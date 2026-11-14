# C1-R G3 LOCAL — Run Report (machine-generated)

- **Verdict:** **PASS**
- v1.1.0; Python 3.13.3; pyarrow 25.0.1; batch 8192 rows; block 256
- Elapsed 128.7 s; peak working set 1249 MiB
- Aggregates only. No sample hash, feature vector or row-level membership record is in this report.

## Feature validation

- cells validated: 4,599,225,741 of expected 4,599,225,741; null 0; non-int8 columns 0; out-of-domain 0; value counts 0/1/2 = 4,524,135,864/75,089,871/6 (accepted domain [0, 1, 2], G3-DEV-01 (Manifest §3.3 amended; §8.4))

## Rows containing value 2 (G3-DEV-01)

| Unit | Label | Partition | cells=2 | seen (exact) | any TRAIN vector has a 2 | descriptive only: seen if 2->1 |
|---|---|---|---|---|---|---|
| FAR_2019 | malware | train | 1 | False | False | False |
| FAR_2020 | benign | train | 1 | False | False | False |
| FAR_2021 | benign | test | 2 | False | False | False |
| FAR_2023 | benign | test | 1 | False | False | False |
| FAR_2024 | benign | train | 1 | False | False | False |

## Estimability (U arm needs >=1 benign and >=1 malware)

| Unit | primary | F benign | F malware | U benign | U malware | U defined |
|---|---|---|---|---|---|---|
| IID_2013 | True | 4,760 | 4,729 | 3,032 | 2,722 | YES |
| IID_2014 | True | 3,545 | 2,252 | 2,437 | 1,782 | YES |
| NEAR_2016 | False | 64,059 | 45,134 | 59,281 | 40,136 | YES |
| NEAR_2017 | False | 77,785 | 21,359 | 76,945 | 20,944 | YES |
| FAR_2018 | True | 64,942 | 39,350 | 64,467 | 38,282 | YES |
| FAR_2019 | True | 49,465 | 41,585 | 48,787 | 39,311 | YES |
| FAR_2020 | True | 55,718 | 46,355 | 55,598 | 45,417 | YES |
| FAR_2021 | True | 45,528 | 35,627 | 45,486 | 35,396 | YES |
| FAR_2022 | True | 44,768 | 41,648 | 44,724 | 40,976 | YES |
| FAR_2023 | True | 46,462 | 7,892 | 46,428 | 7,772 | YES |
| FAR_2024 | True | 47,633 | 794 | 47,622 | 793 | YES |
| FAR_2025 | True | 44,640 | 23 | 44,626 | 23 | YES |

**E2: ESTIMABLE**

