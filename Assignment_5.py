############################################################
#
# Assingment 5
#
#
############################################################
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
from sklearn import mixture


def read_excel_sheet1(excelfile):
    from pandas import read_excel

    return (read_excel(excelfile)).values


def read_excel_range(excelfile, sheetname="Sheet1", startrow=1, endrow=1, startcol=1, endcol=1):
    from pandas import read_excel

    values = (read_excel(excelfile, sheetname, header=None)).values
    return values[startrow-1:endrow, startcol-1:endcol]


def read_excel(excelfile, **args):
    if args:
        data = read_excel_range(excelfile, **args)
    else:
        data = read_excel_sheet1(excelfile)

    if data.shape == (1, 1):
        return data[0, 0]
    elif (data.shape)[0] == 1:
        return data[0]
    else:
        return data


def write_excel_rows(excel_file, sheet_name, start_row, start_column, vector_data):
    write_excel_rows_columns(excel_file, sheet_name, start_row, start_column, vector_data.reshape(1, len(vector_data)))


def write_excel_columns(excel_file, sheet_name, start_row, start_column, vector_data):
    write_excel_rows_columns(excel_file, sheet_name, start_row, start_column, vector_data.reshape(len(vector_data), 1))


def write_excel_rows_columns(excel_file, sheet_name, start_row, start_column, matrix_data):
    wb = openpyxl.load_workbook(excel_file)
    worksheet = wb[sheet_name]

    for i, row in enumerate(matrix_data):
        for j in range(len(row)):

            worksheet.cell(row=start_row + i, column=start_column + j, value=(row[j]).item())

    wb.save(excel_file)


def draw_scatter_plot(data):
    # Draw scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')

    # scatter plot . scatter(x,y, s=the marker size, facecolors=color matrix, marker=symbol's shape."o" means circle )
    ax.scatter(data[:, 0], data[:, 1], s=3, linewidths=0, marker="o");
    ax.set_aspect('equal')

    # plt.gca().invert_yaxis()
    plt.show()


#####################################################################################################
#
#   Main
#
#
######################################################################################################
path=r'C:\Users\uskya\OneDrive\document_usk\UCSC\02_SecondQuater\Introduction_To_MachineLearning_DataMining\Assignment\5_Assignment'
excel_file = path+"\\Assignment_5_Data_and_Template.xlsx"

ORIGINAL_LOAD_FLAG = True

if ORIGINAL_LOAD_FLAG:
    data = read_excel(excelfile=excel_file, sheetname="Data", startrow=2, endrow=951, startcol=1, endcol=2)
else:
    data = np.load(path + "\\training.npy")

# save data at once
# np.save(path+"\\training", data)

data = np.array(data, dtype=float)

# draw_scatter_plot(data)

# Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', tol=0.001, reg_covar=1e-06,
                              max_iter=100, n_init=1, init_params='kmeans', weights_init=None,
                              means_init=None, precisions_init=None, random_state=None, warm_start=False,
                              verbose=0, verbose_interval=10)

gmm.fit(data)

class_array = gmm.predict(data)

prob_array = gmm.predict_proba(data)
data_class = np.column_stack((class_array.T, data))

CLASS_LABELS = ("M", "F", "C")

# get means of classes and add labels
df_class = pd.DataFrame(data_class, columns=["CLASS", "HEIGHT", "HSPAN"])
mean_class = np.array( (df_class.groupby(['CLASS'], as_index=False).mean()).sort_values(by="HEIGHT", ascending=False), dtype=np.str)
mean_class = np.column_stack((CLASS_LABELS, mean_class))

# get the biggest probabilities
row_number = np.arange(0, len(class_array), 1)
prob_biggest = prob_array[row_number, class_array]*100
prob_classified = np.column_stack((row_number, class_array, prob_biggest))

(classified, count) = np.unique(class_array, return_counts=True)

####################################################
# merge Probabilities and Labels by Class Numbers
####################################################
df_prob_classified = pd.DataFrame(prob_classified, columns=["INDEX", "CLASS", "PROB"])
df_class_label = pd.DataFrame(mean_class[:, 0:2], columns=["LABEL", "CLASS"])

# change variable type
df_prob_classified["INDEX"] = df_prob_classified["INDEX"].astype(np.int32)
df_class_label["CLASS"] = df_class_label["CLASS"].astype(np.float64)
df_class_label["LABEL"] = df_class_label["LABEL"].astype(np.str)

# merge
df_labels = pd.merge(df_prob_classified, df_class_label, on='CLASS')
df_exceldata = df_labels.sort_values(by='INDEX').ix[:, ['LABEL', 'PROB']]

# count data
class_label = pd.DataFrame(np.column_stack((np.arange(0, len(CLASS_LABELS), 1), CLASS_LABELS)), columns=["LABEL_INDEX", "LABEL"])
df_exceldata_count = pd.merge(df_exceldata.groupby(['LABEL'], as_index=False).count(), class_label, on='LABEL')
df_exceldata_count = df_exceldata_count.sort_values(by='LABEL_INDEX')

print('df_exceldata_count->{}'.format(df_exceldata_count))

####################################################
# write data to Excel
####################################################
write_excel_columns(excel_file, "Results", 2, 1, np.array(df_exceldata['LABEL'], dtype=np.str))
write_excel_columns(excel_file, "Results", 2, 2, np.array(df_exceldata['PROB'], dtype=np.float))
write_excel_columns(excel_file, "Results", 2, 6, np.array(df_exceldata_count['PROB'], dtype=np.int))
