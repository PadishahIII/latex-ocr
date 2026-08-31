"""
MathJax + CairoSVG Formula Renderer (LaTeX → SVG → PDF)

Replaces Puppeteer with a lighter, faster pipeline:
1. LaTeX → SVG (via MathJax in Node.js)
2. SVG → PDF bytes (via CairoSVG in Python)

Requirements:
    - Node.js (v14+)
    - npm packages: mathjax-full (installed at project root)
    - python packages: cairosvg
    
Installation:
    npm install mathjax-full
    pip install cairosvg

Usage:
    from katex_renderer import KaTeXRenderer
    
    renderer = KaTeXRenderer()
    pdf_bytes, error = renderer.render(r"\\int_0^\\infty e^{-x^2} dx", "output_base")
    if pdf_bytes:
        with open("output.pdf", "wb") as f:
            f.write(pdf_bytes)
"""

import json
import subprocess
import threading
import select
import time
from pathlib import Path
from typing import Optional, Tuple
import re

try:
    import cairosvg
except ImportError:
    cairosvg = None

class MathjaxRenderer:
    """
    Renders LaTeX formulas to PDF bytes using MathJax (Node) + CairoSVG (Python).
    """

    # Project root directory (where node_modules should be installed)
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

    def __init__(
        self,
        output_dir: str | Path | None = None,
        node_path: str = "node",
        timeout: int = 10,
    ):
        """
        Args:
            output_dir: (Unused but kept for API compat)
            node_path: Path to node executable
            timeout: Timeout in seconds for rendering
        """
        if cairosvg is None:
            raise ImportError("Please install cairosvg: pip install cairosvg")

        self.output_dir = Path(output_dir) if output_dir else self._PROJECT_ROOT / "temp_render_katex"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self._lock = threading.Lock()
        self._server_lock = threading.Lock()  # Separate lock for server restart
        self.node_modules_dir = self._PROJECT_ROOT
        self.node_path = node_path
        self.timeout = timeout
        
        # Track consecutive failures for server health monitoring
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3  # Restart server after this many failures
        self._last_restart_time = 0
        self._min_restart_interval = 2.0  # Minimum seconds between restarts

        # Create the Node.js rendering script
        self._setup_renderer_script()
        
        # Start the persistent rendering server
        self._server_process = None
        self._start_server()

    def _start_server(self):
        """Start the persistent Node.js rendering server."""
        import atexit
        import select
        import time

        try:
            self._server_process = subprocess.Popen(
                [self.node_path, str(self.script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.node_modules_dir,
            )
        except Exception as e:
            print(f"Failed to start Node.js process: {e}")
            return

        # Wait for server ready message
        start_time = time.time()
        ready_flag = False
        while time.time() - start_time < 5:
            if self._server_process.stderr:
                ready, _, _ = select.select([self._server_process.stderr], [], [], 0.1)
                if ready:
                    line = self._server_process.stderr.readline()
                    # print(f"Node stderr: {line.strip()}")
                    if "Renderer server ready" in line:
                        ready_flag = True
                        break
        
        if not ready_flag:
            print("Warning: Renderer server did not report ready. Check 'npm install mathjax-full'")

        atexit.register(self._stop_server)
    
    def _stop_server(self):
        """Stop the persistent rendering server."""
        if self._server_process and self._server_process.poll() is None:
            try:
                self._server_process.stdin.close()
            except:
                pass
            try:
                self._server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            if self._server_process.poll() is None:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._server_process.kill()
    
    def _restart_server_if_needed(self, force: bool = False) -> bool:
        """
        Restart the server if it's unhealthy or if force=True.
        Returns True if server is now running, False otherwise.
        """
        current_time = time.time()
        
        # Avoid rapid restarts
        if not force and (current_time - self._last_restart_time) < self._min_restart_interval:
            return self._server_process is not None and self._server_process.poll() is None
        
        with self._server_lock:
            # Double-check after acquiring lock
            if not force and self._server_process and self._server_process.poll() is None:
                return True
                
            print(f"[MathjaxRenderer] Restarting Node.js server (consecutive failures: {self._consecutive_failures})...")
            self._stop_server()
            self._start_server()
            self._last_restart_time = time.time()
            self._consecutive_failures = 0
            
            return self._server_process is not None and self._server_process.poll() is None
    
    def _read_with_timeout(self, timeout: float) -> Optional[str]:
        """
        Read a line from the server's stdout with timeout.
        Returns None if timeout occurs or server is not running.
        """
        if not self._server_process or self._server_process.poll() is not None:
            return None
            
        try:
            # Use select for timeout on Unix
            ready, _, _ = select.select([self._server_process.stdout], [], [], timeout)
            if ready:
                line = self._server_process.stdout.readline()
                return line if line else None
            return None  # Timeout
        except (ValueError, OSError):
            # File descriptor closed or other error
            return None
    
    def __del__(self):
        self._stop_server()

    def _setup_renderer_script(self):
        """Create the Node.js script using MathJax for SVG generation."""
        self.script_path = self.output_dir / "mathjax_svg_server.js"
        
        # This script uses mathjax-full to convert TeX -> SVG string
        script_content = r'''
const readline = require('readline');
let mathjax = null;

async function initMathJax() {
    if (mathjax) return;
    try {
        require('mathjax-full/js/util/entities/all.js');
        const {mathjax: mj} = require('mathjax-full/js/mathjax.js');
        const {TeX} = require('mathjax-full/js/input/tex.js');
        const {SVG} = require('mathjax-full/js/output/svg.js');
        const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
        const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
        const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');

        const adaptor = liteAdaptor();
        RegisterHTMLHandler(adaptor);

        // Configure TeX with error tolerance
        const tex = new TeX({ 
            packages: AllPackages,
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            // Error handling: format errors as red text in the output instead of failing
            formatError: (jax, err) => {
                // Return a node that shows the error visually but continues rendering
                return jax.formatError(err);
            }
        });
        
        const svg = new SVG({ fontCache: 'none' }); // 'none' so paths are self-contained
        
        mathjax = {
            doc: mj.document('', {InputJax: tex, OutputJax: svg}),
            adaptor: adaptor
        };
    } catch (e) {
        console.error("Failed to load mathjax-full. Please run: npm install mathjax-full");
        console.error(e);
        process.exit(1);
    }
}

// Convert formula to SVG string
function renderToSVG(formula, displayMode) {
    if (!mathjax) throw new Error("MathJax not initialized");
    
    // Reset document for fresh conversion
    mathjax.doc.clear();
    
    const node = mathjax.doc.convert(formula, {
        display: displayMode,
        em: 16,
        ex: 8,
        containerWidth: 80 * 16
    });
    
    // Get the SVG HTML string
    const svgHtml = mathjax.adaptor.innerHTML(node);
    
    // Check for MathJax errors in the output
    // MathJax wraps errors in <g data-mml-node="merror" data-mjx-error="...">
    const errorMatch = svgHtml.match(/data-mjx-error="([^"]+)"/);
    if (errorMatch) {
        throw new Error("MathJax: " + errorMatch[1]);
    }
    
    return svgHtml;
}

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false
    });

    await initMathJax();
    console.error('Renderer server ready');

    rl.on('line', (line) => {
        try {
            const request = JSON.parse(line);
            const { formula, options } = request;
            const displayMode = options ? options.displayMode : true;
            
            const svgHtml = renderToSVG(formula, displayMode);
            
            console.log(JSON.stringify({ success: true, svg: svgHtml }));
        } catch (e) {
            console.log(JSON.stringify({ success: false, error: e.message }));
        }
    });
}

main();
'''
        self.script_path.write_text(script_content)

    def _wrap_svg_on_canvas(self, inner_svg: str, padding: int = 10) -> str:
        """
        Wrap the MathJax SVG with a white background rectangle.
        """
        # Clean the inner SVG (remove <?xml...?> if present)
        if inner_svg.startswith("<?xml"):
            inner_svg = inner_svg.split("?>", 1)[1].strip()
        
        # Find the opening <svg> tag
        match = re.search(r'<svg([^>]*)>', inner_svg)
        if not match:
            return inner_svg
        
        svg_attrs = match.group(1)
        svg_tag_end = match.end()
        
        # Extract viewBox to determine dimensions for white rect
        vb_match = re.search(r'viewBox="([^"]+)"', svg_attrs)
        if vb_match:
            vb = vb_match.group(1).split()
            if len(vb) >= 4:
                # viewBox: x, y, width, height
                vb_x, vb_y, vb_w, vb_h = vb[0], vb[1], vb[2], vb[3]
                # Create white background rect covering the viewBox
                white_rect = f'<rect x="{vb_x}" y="{vb_y}" width="{vb_w}" height="{vb_h}" fill="white"/>'
                
                # Insert the rect right after <svg ...>
                # Need to handle potential <defs> - insert rect after defs if present
                defs_end = inner_svg.find('</defs>')
                if defs_end != -1:
                    insert_pos = defs_end + len('</defs>')
                else:
                    insert_pos = svg_tag_end
                
                inner_svg = inner_svg[:insert_pos] + white_rect + inner_svg[insert_pos:]
        
        return inner_svg

    def render(
        self,
        formula: str,
        filename_base: str, # Unused for bytes return
        display_mode: bool = True,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Render LaTeX -> SVG -> PDF Bytes.
        
        Returns:
            (pdf_bytes, error_message)
        """
        # Ensure server is running
        if self._server_process is None or self._server_process.poll() is not None:
            if not self._restart_server_if_needed(force=True):
                return None, "Failed to start rendering server"

        request = {
            "formula": formula,
            "options": {"displayMode": display_mode}
        }

        try:
            with self._lock:
                # Attempt to send request
                try:
                    if self._server_process is None or self._server_process.poll() is not None:
                        raise BrokenPipeError("Server not running")
                    self._server_process.stdin.write(json.dumps(request) + "\n")
                    self._server_process.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    # Server died, try to restart and retry once
                    if not self._restart_server_if_needed(force=True):
                        return None, f"Server restart failed: {e}"
                    try:
                        self._server_process.stdin.write(json.dumps(request) + "\n")
                        self._server_process.stdin.flush()
                    except Exception as e2:
                        return None, f"Failed to send request after restart: {e2}"
                
                # Read response with timeout
                response_line = self._read_with_timeout(self.timeout)
                
                if response_line is None:
                    # Timeout or server died - increment failure counter
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        self._restart_server_if_needed(force=True)
                    return None, f"Server timeout after {self.timeout}s (failure #{self._consecutive_failures})"

            # Parse response
            try:
                result = json.loads(response_line)
            except json.JSONDecodeError as e:
                self._consecutive_failures += 1
                return None, f"Invalid JSON response: {e}"
            
            if not result.get("success"):
                # Don't count MathJax errors as server failures
                return None, result.get("error", "Unknown Render Error")
            
            # Success - reset failure counter
            self._consecutive_failures = 0

            svg_content = result["svg"]
            
            # Add white background to SVG
            svg_content = self._wrap_svg_on_canvas(svg_content)
            
            try:
                # Convert SVG to PDF bytes
                pdf_bytes = cairosvg.svg2pdf(bytestring=svg_content.encode('utf-8'))
                return pdf_bytes, None
            except Exception as e:
                return None, f"CairoSVG Error: {e}"

        except Exception as e:
            self._consecutive_failures += 1
            return None, f"Pipeline Error: {e}"



    def render_to_svg(self, formula: str, display_mode: bool = True) -> Tuple[Optional[str], Optional[str]]:
        """
        Render LaTeX -> SVG string.
        Returns: (svg_content, error_message)
        """
        # Ensure server is running
        if self._server_process is None or self._server_process.poll() is not None:
            if not self._restart_server_if_needed(force=True):
                return None, "Failed to start rendering server"

        request = {
            "formula": formula,
            "options": {"displayMode": display_mode}
        }
        
        try:
            with self._lock:
                # Attempt to send request
                try:
                    if self._server_process is None or self._server_process.poll() is not None:
                        raise BrokenPipeError("Server not running")
                    self._server_process.stdin.write(json.dumps(request) + "\n")
                    self._server_process.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    # Server died, try to restart and retry once
                    if not self._restart_server_if_needed(force=True):
                        return None, f"Server restart failed: {e}"
                    try:
                        self._server_process.stdin.write(json.dumps(request) + "\n")
                        self._server_process.stdin.flush()
                    except Exception as e2:
                        return None, f"Failed to send request after restart: {e2}"
                
                # Read response with timeout
                response_line = self._read_with_timeout(self.timeout)
                
                if response_line is None:
                    # Timeout or server died - increment failure counter
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        self._restart_server_if_needed(force=True)
                    return None, f"Server timeout after {self.timeout}s (failure #{self._consecutive_failures})"

            # Parse response
            try:
                result = json.loads(response_line)
            except json.JSONDecodeError as e:
                self._consecutive_failures += 1
                return None, f"Invalid JSON response: {e}"
            
            if not result.get("success"):
                return None, result.get("error", "Unknown Render Error")
            
            # Success - reset failure counter
            self._consecutive_failures = 0
            
            # Add white background to SVG
            svg_content = self._wrap_svg_on_canvas(result["svg"])
            return svg_content, None
            
        except Exception as e:
            return None, f"Pipeline Error: {e}"

# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MathJax + CairoSVG Renderer Demo")
    print("=" * 60)

    # The problematic formula
    problematic_formula = r"""\begin{array}{rl}{\mathcal{M}_{e,\beta}(\Omega):}&{=\{ m\in\mathcal{M}_{\beta}(\Omega):\textrm{misergodic}\} ,}\\ {\mathcal{M}_{e,\beta}(Y_{u}):}&{=\{ \mu\in\mathcal{M}_{\beta}(Y_{u}):\textrm{\mu isergodic}\} ,}\\ {\mathcal{M}_{\beta,v_{2}}(Y_{u}):}&{=\{ \mu\in\mathcal{M}_{\beta}(Y_{u}):\mu(X_{u}\backslash(X_{u}+v_{2}))=1\} ,}\\ {\mathcal{M}_{e,\beta,v_{2}}(Y_{u}):}&{=\mathcal{M}_{e,\beta}(Y_{u})\cap\mathcal{M}_{\beta,v_{2}}(Y_{u}).}\end{array}"""

    renderer = MathjaxRenderer()

    print("\n1. Rendering to SVG...")
    svg_content, error = renderer.render_to_svg(problematic_formula)
    if svg_content:
        svg_path = Path("test_output.svg")
        svg_path.write_text(svg_content, encoding="utf-8")
        print(f"✅ SVG saved to: {svg_path.absolute()}")
    else:
        print(f"❌ SVG Failed! Error: {error}")

    print("\n2. Rendering to PDF...")
    pdf_bytes, error = renderer.render(problematic_formula, "test")
    
    if pdf_bytes:
        pdf_path = Path("test_output.pdf")
        pdf_path.write_bytes(pdf_bytes)
        print(f"✅ PDF saved to: {pdf_path.absolute()}")
    else:
        print(f"❌ PDF Failed! Error: {error}")
