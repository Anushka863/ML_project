# PTB-XL Target Summary

## TARGET DEFINITION
**Positive class (1)**: Abnormal ECG (all other diagnostic classes)
**Negative class (0)**: Normal ECG (NORM)
**Reason**: To create a robust, binary baseline proof-of-concept for detecting cardiac abnormalities before mapping granular classes to complex NHANES equivalents.
**Source label(s)**: `scp_codes` dict containing 'NORM'.

## STATISTICS
* **Number of patients**: 18869
* **Number of ECG records**: 21799
* **Normal count (Negative)**: 9514
* **Abnormal count (Positive)**: 12285

## CLASS BALANCE
* **Normal %**: 43.64%
* **Abnormal %**: 56.36%
