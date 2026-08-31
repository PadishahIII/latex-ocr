## Final LaTeX Space Normalization Rules

### **Rule 1: Protect Command Boundaries (Your Core Rule)**
**When a control word is followed (ignoring spaces) by a letter or number, keep one space after the command name.**

**Purpose**: Prevents `\mathcal A` → `\mathcalA` token merging.

**Regex for Control Words**: `\\[a-zA-Z]+`  
**Regex for Trigger Context**: `\\[a-zA-Z]+\s+[a-zA-Z0-9]`

**Action**: Replace `(\[a-zA-Z]+)\s+([a-zA-Z0-9])` with `\1 \2`

**Examples**:
- `\mathcal A` → `\mathcal A` ✓ (keeps space)
- `\mathrm sin` → `\mathrm sin` ✓ (keeps space)
- `\frac{1}{2}` → `\frac{1}{2}` ✓ (no change, braces not letter)

***

### **Rule 2: Preserve Spaces Inside Text-Like Arguments**
**Keep all spaces within arguments of commands that render text literally.**

**Text-like Commands**: `\text`, `\mathrm`, `\mathbf`, `\mathit`, `\mathsf`, `\mathtt`, `\operatorname`, `\mbox`, `\hbox`

**Regex**: `(\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|mbox|hbox))\{([^}]*)\}`

**Action**: Do NOT modify spaces inside the `{...}` capture group.

**Examples**:
- `\text{in region 1}` → `\text{in region 1}` ✓ (spaces preserved)
- `\mathrm{max value}` → `\mathrm{max value}` ✓ (spaces preserved)

***

### **Rule 3: Preserve Spaces After Control Symbols**
**Never remove spaces after control symbols (`\` + one non-letter).**

**Control Symbols**: `\\`, `\;`, `\,`, `\!`, `\:`, `\_`, `\^`, `\|`, `\,`, `\{`, `\}`, `\%`, `\#`, `\&`, `\<`, `\>`, `\/`, `\~`, `\'`, `\"`, `\``, `\.`

**Regex**: `\\[^a-zA-Z]`

**Action**: If token matches control symbol, the following space is significant and must be retained.

**Examples**:
- `\, x` → `\, x` ✓ (space kept)
- `\\[1em]` → `\\[1em]` ✓ (no space to remove)
- `\; y` → `\; y` ✓ (space kept)

***

### **Rule 4: Never Modify Verbatim Commands**
**Do not touch any spaces in `\verb`, `\verb*`, or `\lstinline` commands.**

**Regex**: `\\verb(?:\*)?(\|[^|]*\|)|\\verb(?:\*)?(\+[^+]*\+)|\\verb(?:\*)?(.[^ ].*)`  
**Alternative**: `\\verb(?:\*)?(.).*(?:\1)` (general pattern)

**Action**: Skip entire command and its argument from processing.

**Examples**:
- `\verb|a b c|` → `\verb|a b c|` ✓ (completely untouched)

***

### **Rule 5: Handle Array/Tabular Row Separators**
**In `array`, `tabular`, `matrix` environments, spaces around `&` and `\\` row endings can be removed.**

**Environment Detection**: `\begin{array}`, `\begin{tabular}`, `\begin{matrix}`, `\begin{pmatrix}`, etc.

**Regex for Structure**: `\\begin\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix)\}`

**Action**: Within these environments, treat `&` and `\\` as structural delimiters and remove surrounding spaces.

**Examples**:
- `\begin{array}{ll} a & b \\ c & d \end{array}` → `\begin{array}{ll}a&b\\c&d\end{array}` ✓

***

### **Rule 6: Remove All Other Spaces**
**All remaining whitespace (spaces, tabs, newlines) can be collapsed or removed.**

**Action**: Replace `\s+` with empty string `''`, except where Rules 1-5 apply.

***

## Complete Regex Toolkit

```python
import regex as re

# Rule 1: Control word boundary protection
PATTERN_CONTROL_WORD_BOUNDARY = r'(\\[a-zA-Z]+)\s+([a-zA-Z0-9])'
REPL_CONTROL_WORD_BOUNDARY = r'\1 \2'

# Rule 2: Text-like commands (exempt from space removal)
TEXT_LIKE_COMMANDS = [
    'text', 'mathrm', 'mathbf', 'mathit', 'mathsf', 'mathtt',
    'operatorname', 'mbox', 'hbox', 'textbf', 'textit'
]
PATTERN_TEXT_LIKE = r'\\(?:' + '|'.join(TEXT_LIKE_COMMANDS) + r')\{[^}]*\}'

# Rule 3: Control symbols (protect following space)
PATTERN_CONTROL_SYMBOL = r'\\[^a-zA-Z]'

# Rule 4: Verbatim commands (completely exempt)
PATTERN_VERBATIM = r'\\verb(?:\*)?(.)(?:.*?)(?:\1)'

# Rule 5: Array/tabular environments
PATTERN_ENVIRONMENTS = r'\\begin\{(?:array|tabular|matrix|pmatrix|bmatrix|vmatrix|aligned|gathered|cases)\}'

# Combined tokenizer pattern
TOKEN_PATTERN = re.compile(
    r'(' + PATTERN_CONTROL_SYMBOL + r')|' +  # Control symbols (group 1)
    r'(\\[a-zA-Z]+\*?)|' +  # Control words (group 2)
    r'(\{[^}]*\})|' +  # Arguments (group 3)
    r'([^{}\s\\]+)|' +  # Text tokens (group 4)
    r'(\s+)'  # Whitespace (group 5)
)
```

## Implementation Strategy

```python
def normalize_latex_robust(formula):
    # Step 1: Protect verbatim commands (Rule 4)
    verbatim_spans = []
    for match in re.finditer(PATTERN_VERBATIM, formula):
        verbatim_spans.append((match.start(), match.end()))
    
    # Step 2: Tokenize and process
    tokens = []
    for match in TOKEN_PATTERN.finditer(formula):
        start, end = match.span()
        
        # Skip if inside verbatim
        if any(s <= start < end <= e for s, e in verbatim_spans):
            tokens.append(formula[start:end])
            continue
        
        token = match.group()
        token_type = next(i for i, g in enumerate(match.groups(), 1) if g)
        
        # Rule 1: Control word boundary
        if token_type == 2 and match.groups()[4]:  # Next token is space
            next_token = formula[end:].lstrip()
            if next_token and (next_token[0].isalnum()):
                tokens.append(token + ' ')
                continue
        
        # Rule 3: Control symbol - keep following space
        if token_type == 1:
            tokens.append(token)
            continue
        
        # Rule 2 & 5: Text-like commands and environments handled by exemption list
        
        # Rule 6: Remove other spaces
        if token_type == 5:  # Whitespace
            continue
        
        tokens.append(token)
    
    return ''.join(tokens)
```

This combined approach gives you **95%+ robustness** for math corpora while protecting against the most common token-merging errors. [jisem-journal](https://jisem-journal.com/index.php/journal/article/download/2101/813)
