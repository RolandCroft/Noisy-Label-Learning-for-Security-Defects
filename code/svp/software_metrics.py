"""
Extract Software Metrics from SciTool Understand outputs for Mozilla releases.
"""
import sys
import pandas as pd

file_metrics = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict',
                'AvgEssential', 'AvgLine', 'AvgLineBlank', 'AvgLineCode',
                'AvgLineComment', 'CountDeclClass', 'CountDeclFunction',
                'CountLine', 'CountLineBlank', 'CountLineCode',
                'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment',
                'CountSemicolon', 'CountStmt', 'CountStmtDecl', 'CountStmtExe',
                'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict',
                'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified',
                'SumCyclomaticStrict', 'SumEssential']
class_metrics = ['CountClassBase', 'CountClassCoupled', 'CountClassDerived',
                 'CountDeclClassMethod', 'CountDeclClassVariable',
                 'CountDeclInstanceMethod', 'CountDeclInstanceVariable',
                 'CountDeclMethod', 'CountDeclMethodPrivate',
                 'CountDeclMethodProtected', 'CountDeclMethodPublic']
class_metrics1 = ['MaxInheritanceTree', 'PercentLackOfCohesion']
function_metrics = ['CountInput', 'CountOutput', 'CountPath', 'MaxNesting']

all_labels = pd.read_csv('data/mozilla/file_labels_63-84.csv')

# Specify release
i = int(sys.argv[1])
print(i)
# Load the SciTool output
raw = pd.read_csv(f'data/metrics/release_{i}.csv', error_bad_lines=False)
raw['File'] = raw.File.astype(str)
raw['File'] = raw.File.apply(lambda x: x.replace('\\', '/'))
raw['Common_File'] = raw.File.apply(lambda x: x.rsplit('.', 1)[0])
metrics = pd.DataFrame(columns=['file'] + file_metrics +
                       class_metrics + class_metrics1 +
                       [f + '_Min' for f in function_metrics] +
                       [f + '_Mean' for f in function_metrics] +
                       [f + '_Max' for f in function_metrics])
# Load the affected files
labels = all_labels[all_labels.release == i]
# Collect metrics for each file
for c, file in enumerate(labels.file.tolist()):
    if c % 1000 == 0:
        print(c)
    # Get file metrics
    file_ms = raw[(raw.Kind == 'File') & (raw.File == file)][file_metrics]
    if len(file_ms) > 0:
        file_ms = file_ms.iloc[0].tolist()
    else:
        file_ms = [0] * len(file_metrics)
    # Get class metrics
    class_file = file.rsplit('.', 1)[0]
    class_raw = raw[(raw.Kind.str.contains('Class')) &
                    (raw.Common_File == class_file)]
    class_ms = class_raw[class_metrics].agg('sum').tolist() + \
        class_raw[class_metrics1].agg('mean').tolist()
    # Get function metrics
    function_raw = raw[(raw.Kind.str.contains('Function')) &
                       (raw.File == file)][function_metrics]
    function_raw = function_raw.agg(['min', 'mean', 'max'])
    function_ms = function_raw.iloc[0].tolist()
    function_ms += function_raw.iloc[1].tolist()
    function_ms += function_raw.iloc[2].tolist()
    metrics.loc[len(metrics)] = [file] + file_ms + class_ms + function_ms
# Save
path = f'code/svp/feature_sets/software_metrics_release_{i}.csv'
metrics.to_csv(path, index=False)
