# Portfolio migration

The source snapshot is commit `983ecbfbde726bbf38d2d86ddf1d576c1361c4e1`.
Original notebooks remain recoverable in Git history. The new notebooks are rebuilt around shared,
tested code; old stored metrics are not carried forward as evidence for the new implementations.

| Previous location | New project |
| --- | --- |
| Linear Regression/Car Price Prediction | projects/car-price |
| Decision Tree/Medical Insurance Cost Prediction | projects/insurance-cost |
| kNN/Heart Disease Prediction | projects/heart-disease |
| Logistic Regression/Student Pass-Fail Prediction | projects/student-grades |
| Naive Bayes/SMS Spam Detection | projects/sms-spam |
| SVM/Breast Cancer Prediction | projects/breast-cancer |
| SVM/California Housing Price Prediction | projects/california-housing |

The loan-approval entry was an unresolved Git submodule pointer with no `.gitmodules` mapping
or checked-in implementation. It is removed from the active project inventory; no loan model
has been fabricated. Editor checkpoints, virtual documents, bytecode, and obsolete notebook
builders are removed. Six source CSVs are moved without content changes; duplicate checkpoint CSVs
are editor artifacts and are removed.

Methodological corrections include excluding GPA, removing repeated heart records before splitting,
using training-only categorical preprocessing, replacing deprecated metric/model arguments, and
removing the incompatible housing fallback and incorrect ratio features. Search spaces are bounded
and use three folds to keep the portfolio practical to reproduce. Housing now focuses on SVR versus
a dummy baseline; the original multi-model exploration remains in history.
