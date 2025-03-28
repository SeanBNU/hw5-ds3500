# Getting Started

sean causing fuckery

Welcome to the repo! This guide will help you set up the project locally. Make sure to open or navigate to the directory where you'd like the repo folder to appear.

---

## Cloning the Repo

### Method 1: VSCode / IDE

1. Open a fresh VSCode window and click "clone from GitHub URL" or launch the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Select **"Git: Clone"**.
3. Paste the URL:  
   ```
   https://github.com/SeanBNU/hw5-ds3500
   ```
4. Choose your destination folder and open the repo.

### Method 2: Terminal

1. Open your terminal and navigate to your desired folder:
   ```bash
   cd /path/to/your/directory
   ```
2. Clone the repo and navigate into it:
   ```bash
   git clone    https://github.com/SeanBNU/hw5-ds3500.git
   cd hw5-ds3500
   ```

---

## Setup

Install Git LFS and pull any large files:

```bash
git lfs install
git lfs pull
```

---

## Workflow

- **Branching:** Always create a new branch for your edits:
  ```bash
  git checkout -b your-feature-name
  ```

- **Push:** Push your branch when you're ready:
  ```bash
  git push origin your-feature-name
  ```

- **Pull Request:** Merge your branch into `main` only after thorough testing.

---

## Note on the `ai` Branch

The `ai` branch contains a one-shot solution from `o3-mini-high` that can serve as inspiration. To check it out:

```bash
git checkout ai
```

---
