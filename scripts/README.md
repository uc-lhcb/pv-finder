# Processing data

You should "gen"erate `pv_*.root` files and "ana"lyze your data first to `kernel_*.root`. Then, you need to produce collected and prepared kernel files:

```bash
./ProcessFiles.py -o data/Oct03_20K_val.h5    /data/schreihf/PvFinder/kernel_20181003_{1,2}.root
./ProcessFiles.py -o data/Oct03_20K_test.h5   /data/schreihf/PvFinder/kernel_20181003_{3,4}.root
./ProcessFiles.py -o data/Oct03_40K_train.h5  /data/schreihf/PvFinder/kernel_20181003_{5,6,7,8}.root
./ProcessFiles.py -o data/Oct03_80K_train.h5  /data/schreihf/PvFinder/kernel_20181003_{9,10,11,12,13,14,15,16}.root
./ProcessFiles.py -o data/Oct03_80K2_train.h5 /data/schreihf/PvFinder/kernel_20181003_{17,18,19,20,21,22,23,24}.root
./ProcessFiles.py -o data/Aug14_80K_train.h5  /data/schreihf/PvFinder/kernel_20180814_{1,2,3,4,5,6,7,8}.root
```

The current files are:

|        From       |          To         |         Events          |
|-------------------|---------------------|-------------------------|
| `kernel_20181003` | `Oct03_20K_val`     | 1,2                     |
| `kernel_20181003` | `Oct03_20K_test`    | 3,4                     |
| `kernel_20181003` | `Oct03_40K_train`   | 5,6,7,8                 |
| `kernel_20181003` | `Oct03_80K_train`   | 9,10,11,12,13,14,15,16  |
| `kernel_20181003` | `Oct03_80K2_train`  | 17,18,19,20,21,22,23,24 |
| `kernel_20180814` | `Aug14_80K_train`   | 1,2,3,4,5,6,7,8         |

It can take about 40 seconds to save an 80K file.