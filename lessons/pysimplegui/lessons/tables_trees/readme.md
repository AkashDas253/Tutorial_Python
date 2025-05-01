## Tables and Trees in PySimpleGUI

PySimpleGUI supports two structured data display elements: **Tables** for grid-like data and **Trees** for hierarchical data. Both provide ways to organize and interact with structured datasets within the GUI.

---

## Tables

Tables display data in rows and columns, ideal for tabular datasets like spreadsheets, CSVs, or database records.

### Element: `sg.Table`

#### Syntax

```python
sg.Table(
    values,                 # List of lists (rows of data)
    headings=None,          # List of column headers
    visible_column_map=None,# List of booleans to show/hide columns
    col_widths=None,        # List of column widths
    auto_size_columns=True, # Automatically size columns
    display_row_numbers=False, # Show row numbers
    justification='right', # Text alignment
    num_rows=10,            # Number of rows visible
    alternating_row_color=None, # Alternate row background
    selected_row_colors=None,   # Row highlight color
    enable_events=False,    # Triggers event on row selection
    bind_return_key=False,  # Submit on Enter
    key=None,               # Unique key
    tooltip=None            # Tooltip on hover
)
```

#### Example

```python
import PySimpleGUI as sg

data = [[f'Row {i} Col {j}' for j in range(3)] for i in range(10)]
layout = [[sg.Table(values=data, headings=['Col1', 'Col2', 'Col3'], key='-TABLE-', enable_events=True)]]
window = sg.Window('Table Example', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '-TABLE-':
        print(f"Selected row: {values['-TABLE-']}")

window.close()
```

#### Key Features

- **Selectable Rows**: Can return index(es) of selected rows.
- **Column Customization**: Width, visibility, alignment.
- **Colors**: Alternating row and selected row coloring.
- **Non-editable**: Tables are read-only; use `Input` or `Multiline` for editable cells.

---

## Trees

Trees display hierarchical data (e.g., folders, XML, organization charts).

### Element: `sg.Tree`

#### Syntax

```python
sg.Tree(
    data,                   # TreeData instance
    headings=[],            # Column headers
    auto_size_columns=True, # Auto column width
    num_rows=10,            # Visible rows
    col_widths=None,        # List of column widths
    show_expanded=False,    # Expand all nodes initially
    justification='right',  # Text alignment
    select_mode=sg.TABLE_SELECT_MODE_BROWSE, # Selection type
    enable_events=False,    # Triggers event on selection
    key=None,               # Unique key
    tooltip=None            # Tooltip on hover
)
```

### `TreeData` Structure

```python
tree_data = sg.TreeData()
tree_data.insert('', 'root', 'Root Node', ['Column 1'])
tree_data.insert('root', 'child1', 'Child 1', ['C1 Data'])
```

- **Parent Key**: The parent node key (empty string for root).
- **Node Key**: Unique ID for the node.
- **Text**: Display label for the node.
- **Values**: List of values for each column (besides text).

#### Example

```python
import PySimpleGUI as sg

tree_data = sg.TreeData()
tree_data.insert('', 'A', 'Folder A', ['A data'])
tree_data.insert('A', 'A1', 'Subfolder A1', ['A1 data'])
tree_data.insert('', 'B', 'Folder B', ['B data'])

layout = [[sg.Tree(data=tree_data, headings=['Details'], auto_size_columns=True, num_rows=10, key='-TREE-', enable_events=True)]]
window = sg.Window('Tree Example', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '-TREE-':
        print(f"Selected node: {values['-TREE-']}")

window.close()
```

---

## Comparison Table

| Feature            | `sg.Table`                    | `sg.Tree`                             |
|--------------------|-------------------------------|----------------------------------------|
| Structure          | Tabular (rows × columns)      | Hierarchical (nodes with children)     |
| Editable           | No                            | No                                     |
| Multiple Columns   | Yes                           | Yes (besides tree label column)        |
| Selection          | Row-based                     | Node-based                             |
| Events             | On selection                  | On selection                           |
| Use Cases          | Data frames, CSV, reports     | Filesystems, org charts, JSON viewers  |

---

## Notes

- Tables and Trees are **not editable**, but their selections can be used to trigger updates to other GUI elements.
- For **real-time updates**, call `update()` on the element with new values or structure.
- Tree data requires hierarchical consistency; every child node must reference a valid parent.
- Use `enable_events=True` to receive user interactions in the event loop.
