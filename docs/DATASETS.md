# Dataset register

Six raw CSVs are inherited byte-for-byte from commit
`983ecbfbde726bbf38d2d86ddf1d576c1361c4e1`. The earlier repository did not establish source URLs,
licenses, collection dates, or consent details. Familiar filenames are not sufficient proof of
provenance. **Source and redistribution rights remain unverified** for these six files.

| Project | File under its data/raw folder | Target | Provenance status |
| --- | --- | --- | --- |
| Car price | cars.csv | price | Inherited CSV; original source unverified |
| Insurance cost | insurance.csv | charges | Inherited CSV; original source unverified |
| Heart disease | heart.csv | target | Inherited CSV; label dictionary unverified |
| Student grades | students.csv | GradeClass | Inherited CSV; original source unverified |
| SMS spam | spam.csv | v1 → label | Inherited Latin-1 CSV; original source unverified |
| Breast cancer | breast_cancer.csv | diagnosis | Inherited CSV; original source unverified |
| California housing | california_housing.csv | MedHouseVal | Explicit scikit-learn loader; locally cached |

The housing downloader calls `sklearn.datasets.fetch_california_housing(as_frame=True)` and
requires exactly its nine-column schema. Consult the loader's returned `DESCR` for source details
and upstream terms before reuse. The downloaded CSV and loader cache are ignored by Git.
The results report records its SHA-256 fingerprint along with all other datasets.

Before client reuse, record the original landing page and publisher, version, collection method,
license or written permission, feature dictionary, target definition, and population limitations.
Verify that each feature exists at prediction time. Do not infer permissions from a public repository.
