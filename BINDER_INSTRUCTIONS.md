# MyBinder launch instructions

1. Upload this repository to a **public GitHub repository**.
2. Open https://mybinder.org/
3. Paste the GitHub repository URL.
4. Set **Git ref** to `main` while testing. For the journal version, use a fixed release/tag such as `v1.0`.
5. Under **Path to a notebook file**, enter:

   `Journal_Reproducibility.ipynb`

6. Click **Launch** and use **Run All** in the notebook.
7. Confirm that the pipeline and verification cells both complete successfully.
8. Copy the generated Binder URL into the README and manuscript Code Availability section.

## Suggested Binder URL pattern

`https://mybinder.org/v2/gh/<GITHUB_USERNAME>/<REPOSITORY>/v1.0?filepath=Journal_Reproducibility.ipynb`

Replace the placeholders only after the GitHub repository and release exist.
