## `conda` – Python Package and Environment Manager

---

### What is `conda`?

`conda` is an **open-source, cross-platform package and environment manager** that can manage **Python and non-Python dependencies** (like C libraries).

* Comes with **Anaconda** and **Miniconda**
* Can manage environments **and** packages

---

### Package Management

#### Common Commands

| Command                     | Description                          |
| --------------------------- | ------------------------------------ |
| `conda install <pkg>`       | Install a package                    |
| `conda install <pkg>=1.2.3` | Install specific version             |
| `conda update <pkg>`        | Update package                       |
| `conda remove <pkg>`        | Uninstall a package                  |
| `conda list`                | List all installed packages          |
| `conda search <pkg>`        | Search package in conda repositories |
| `conda info`                | Show info about conda setup          |
| `conda clean --all`         | Clear index cache, logs, etc.        |

---

### Channels

* Default: `defaults`
* Community: `conda-forge`, `bioconda`, etc.
* Use:

  ```bash
  conda install -c conda-forge <pkg>
  ```

---

### Configuration Files

| File       | Purpose                                |
| ---------- | -------------------------------------- |
| `.condarc` | Config file (channels, env dirs, etc.) |

---

### Environment Management

#### Environment Commands

| Command                            | Description                    |
| ---------------------------------- | ------------------------------ |
| `conda create -n <env> python=3.9` | Create new environment         |
| `conda activate <env>`             | Activate environment           |
| `conda deactivate`                 | Deactivate current environment |
| `conda env list`                   | List all environments          |
| `conda remove -n <env> --all`      | Delete environment             |
| `conda list --name <env>`          | List packages in specific env  |

---

### Export & Recreate Environments

```bash
conda env export > environment.yml     # Export environment
conda env create -f environment.yml    # Recreate from file
```

---

### Comparing `conda` vs `pip`

| Feature              | `conda`        | `pip`                                   |
| -------------------- | -------------- | --------------------------------------- |
| Binary support       | ✅ Yes          | ❌ No                                    |
| Manages environments | ✅ Yes          | ❌ No (needs `venv`)                     |
| Installs non-Py libs | ✅ Yes          | ❌ No                                    |
| Source               | conda channels | PyPI                                    |
| Speed & stability    | Often faster   | Sometimes slower due to building wheels |

---

### `Miniconda` vs `Anaconda`

| Feature               | Miniconda   | Anaconda                   |
| --------------------- | ----------- | -------------------------- |
| Size                  | Lightweight | Full suite                 |
| Preinstalled packages | Minimal     | 250+ data science packages |
| Flexibility           | High        | Less, but convenient       |

---
