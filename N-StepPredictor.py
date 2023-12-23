#
#
#                     _   _        _____ _              ______             _ _      _
#                    | \ | |      /  ___| |             | ___ \           | (_)    | |
#                    |  \| |______\ `--.| |_ ___ _ __   | |_/ / __ ___  __| |_  ___| |_ ___  _ __
#                    | . ` |______|`--. \ __/ _ \ '_ \  |  __/ '__/ _ \/ _` | |/ __| __/ _ \| '__|
#                    | |\  |      /\__/ / ||  __/ |_) | | |  | | |  __/ (_| | | (__| || (_) | |
#                    \_| \_/      \____/ \__\___| .__/  \_|  |_|  \___|\__,_|_|\___|\__\___/|_|
#                                               | |
#                                               |_|
#                                                                            twitter: 0xihsn
#
#


import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from tkinter import scrolledtext, messagebox, filedialog
from os import path
import math
from tkinter import *
from tkinter.ttk import *
import tkinter.font

# set seed
np.random.seed(7)

# initial the global variables
data = None
train_train = []
train_test = []
train = []
x_train = []
inputs_nmbr = 1
inputs_indices = []
outputs_nmbr = 2
outputs_indices = []
step_number = 0
first_values = []
UI = None
mdl = None
min_neuron = 1
max_neuron = 5
last_input_indices = []

# initialize the fields check arrays
model_is_ready_to_train = [False, False, False]
model_is_ready_to_predict = [False, False, False]


class AppUI(Tk):
    def __init__(self):
        Tk.__init__(self)

        dFont = tkinter.font.Font(family="Arial", size=8)

        # The first row
        self.fm1 = Frame(self, relief=RAISED, borderwidth=1)
        self.dataset_lbl = Label(self.fm1, text="Dataset : ", font=("Arial Bold", 8))
        self.dataset_lbl.pack(side=LEFT, padx=5, pady=5)
        self.path_btn = Button(self.fm1, text="Open", command=get_data_set)
        self.path_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm1.pack(fill=BOTH, side=TOP, expand=True)

        # The second row
        self.fm2 = Frame(self, relief=RAISED, borderwidth=1)
        self.min_nrn_lbl = Label(self.fm2, text="Min Neuron : ", font=("Arial Bold", 8))
        self.min_nrn_lbl.pack(side=LEFT, padx=5, pady=5)
        self.min_nrn_entry = Entry(self.fm2, width=5)
        self.min_nrn_entry.pack(side=LEFT, padx=5, pady=5)
        self.max_nrn_lbl = Label(self.fm2, text="Max Neuron : ", font=("Arial Bold", 8))
        self.max_nrn_lbl.pack(side=LEFT, padx=5, pady=5)
        self.max_nrn_entry = Entry(self.fm2, width=5)
        self.max_nrn_entry.pack(side=LEFT, padx=5, pady=5)
        self.set_neurons_btn = Button(self.fm2, text="set", command=self.set_neurons)
        self.set_neurons_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm2.pack(fill=BOTH, side=TOP, expand=True)

        # The third row
        self.fm3 = Frame(self, relief=RAISED, borderwidth=1)
        self.inputs_nmbr_lbl = Label(self.fm3, text="indices : ", font=("Arial Bold", 8))
        self.inputs_nmbr_lbl.pack(side=LEFT, padx=5, pady=5)
        self.inputs_nmbr_lbl = Label(self.fm3, text="input: ", font=("Arial", 8))
        self.inputs_nmbr_lbl.pack(side=LEFT, padx=5, pady=5)
        self.inputs_nmbr_entry = Entry(self.fm3, width=10)
        self.inputs_nmbr_entry.insert(END, '1,2,3')
        self.inputs_nmbr_entry.pack(side=LEFT, padx=5, pady=5)
        self.outputs_nmbr_lbl = Label(self.fm3, text="target: ", font=("Arial", 8))
        self.outputs_nmbr_lbl.pack(side=LEFT, padx=5, pady=5)
        self.outputs_nmbr_entry = Entry(self.fm3, width=10)
        self.outputs_nmbr_entry.insert(END, '4,5,6')
        self.outputs_nmbr_entry.pack(side=LEFT, padx=5, pady=5)
        self.set_indices_btn = Button(self.fm3, text="set", command=self.set_indices)
        self.set_indices_btn.pack(side=RIGHT, padx=5, pady=5)
        self.inputs_nmbr_progress_txt = Label(self.fm3, text="", font=("Arial Bold", 9))
        self.inputs_nmbr_progress_txt.pack(side=RIGHT, padx=5, pady=5)
        self.fm3.pack(fill=BOTH, side=TOP, expand=True)

        # The fourth row
        self.fm4 = Frame(self, relief=RAISED, borderwidth=1)
        self.progress = Progressbar(self.fm4, orient='horizontal', length=200, mode='determinate')
        self.progress.pack(side=LEFT, fill=X, padx=3, pady=3)
        self.train_btn = Button(self.fm4, text="Train", command=check_train_fields)
        self.train_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm4.pack(fill=BOTH, side=TOP, expand=True)

        # The fifth row
        self.fm5 = Frame(self, relief=RAISED, borderwidth=1)
        self.step_num_lbl = Label(self.fm5, text="how many step do you want to predict : ", font=("Arial Bold", 8))
        self.step_num_lbl.pack(side=LEFT, padx=5, pady=5)
        self.set_step_btn = Button(self.fm5, text="set", command=self.set_steps)
        self.set_step_btn.pack(side=RIGHT, padx=5, pady=5)
        self.step_num_entry = Entry(self.fm5, width=5)
        self.step_num_entry.pack(side=RIGHT, padx=5, pady=5)
        self.fm5.pack(fill=BOTH, side=TOP, expand=True)

        # The sixth row
        self.fm6 = Frame(self, relief=RAISED, borderwidth=1)
        self.first_values_lbl = Label(self.fm6, text="Your first value : ", font=("Arial Bold", 8))
        self.first_values_lbl.pack(side=LEFT, padx=5, pady=5)
        self.first_values_btn = Button(self.fm6, text="set", command=self.set_first_values)
        self.first_values_btn.pack(side=RIGHT, padx=5, pady=5)
        self.first_values_entry = Entry(self.fm6, width=10)
        self.first_values_entry.pack(side=RIGHT, padx=5, pady=5)
        self.fm6.pack(fill=BOTH, side=TOP, expand=True)

        # The seventh row
        self.fm7 = Frame(self, relief=RAISED, borderwidth=1)
        self.predict_btn = Button(self.fm7, text="Predict", command=check_predict_fields)
        self.predict_btn.pack(side=RIGHT, padx=5, pady=5)
        self.fm7.pack(fill=BOTH, side=TOP, expand=True)

        # The eighth row
        self.fm8 = Frame(self, relief=RAISED, borderwidth=1)
        self.results_dump = scrolledtext.ScrolledText(self.fm8, width=30, height=15, font=dFont)
        self.results_dump.pack(fill=BOTH, padx=3, pady=3)
        self.fm8.pack(fill=BOTH, side=TOP, expand=True)

    global model_is_ready_to_train

    # the neurons setting function
    def set_neurons(self):
        global min_neuron, max_neuron
        minstr = self.min_nrn_entry.get()
        maxstr = self.max_nrn_entry.get()
        min_neuron = int(minstr) if isitnumber(minstr) is True else 0
        max_neuron = int(maxstr) if isitnumber(maxstr) is True else 0

        if min_neuron >= max_neuron:
            messagebox.showinfo('Error: Wrong Value',
                                "- the values must be a number \n"
                                "- minimum value can't be equal or bigger then maximum value")
        else:
            model_is_ready_to_train[1] = True

    # FUNCTION FOR LETTING THE USER SET MORE THEN ONE LAG
    def set_indices(self):

        global inputs_nmbr, inputs_indices, outputs_nmbr, outputs_indices
        inputs_indices = []
        outputs_indices = []
        model_is_ready_to_train[2] = True

        inputs_nmbr_tmp = self.inputs_nmbr_entry.get()
        if (inputs_nmbr_tmp[0] == ",") or (inputs_nmbr_tmp[len(inputs_nmbr_tmp)-1] == ","):
            model_is_ready_to_train[2] = False
        inputs_nmbr = 0
        for x in inputs_nmbr_tmp.split(","):
            if x.isdigit():
                inputs_indices.append(int(x))
                inputs_nmbr += 1
            else:
                model_is_ready_to_train[2] = False
                messagebox.showinfo('Error: Wrong Value',
                                    "- please enter correct indices for 'input' values\n"
                                    "  Ex: 2,5,7,10")
                break

        outputs_nmbr_tmp = self.outputs_nmbr_entry.get()
        if (outputs_nmbr_tmp[0] == ",") or (outputs_nmbr_tmp[len(outputs_nmbr_tmp) - 1] == ","):
            model_is_ready_to_train[2] = False
        outputs_nmbr = 0
        for y in outputs_nmbr_tmp.split(","):
            if y.isdigit():
                outputs_indices.append(int(y))
                outputs_nmbr += 1
            else:
                model_is_ready_to_train[2] = False
                messagebox.showinfo('Error: Wrong Value',
                                    "- please enter correct indices for 'output' values\n"
                                    "  Ex: 2,5,7,10")
                break

        if inputs_indices is [] or outputs_indices is []:
            model_is_ready_to_train[2] = False

        if (data is not None) and (inputs_nmbr is outputs_nmbr):
            acl_data_x, acl_data_y = prepare_data(train, inputs_indices, outputs_indices)
            self.inputs_nmbr_progress_txt.configure(text="Autocorrelation is {}"
                                                    .format(round(acl_func(acl_data_x, outputs_indices[0]), 3)))
            self.update()
        else:
            self.inputs_nmbr_progress_txt.configure(text="")

    def set_steps(self):
        global step_number, model_is_ready_to_predict
        steptmp = self.step_num_entry.get()
        step_number = int(steptmp) if isitnumber(steptmp) is True else None
        step_number += 1

        if step_number:
            model_is_ready_to_predict[1] = True

    def set_first_values(self):
        global first_values, model_is_ready_to_predict
        frsttmp = self.first_values_entry.get()
        model_is_ready_to_predict[2] = True
        first_values = []
        for x in frsttmp.split(","):
            if x.isdigit():
                first_values.append(int(x))
            else:
                model_is_ready_to_predict[2] = False
                messagebox.showinfo('Error: Wrong Value',
                                    "- you have to enter " + str(inputs_nmbr) + " numbers seperated by (,)\n")
                break
        if len(first_values) is not inputs_nmbr:
            model_is_ready_to_predict[2] = False
            first_values = []
            messagebox.showinfo('Error: Wrong Value',
                                "- you have to enter " + str(inputs_nmbr) + " input seperated by (,)\n")


def get_data_set():
    global train_train, train_test, train, data, data_length, model_is_ready_to_train
    data_set_path = filedialog.askopenfilename(initialdir=path.dirname(__file__))

    if data_set_path is not '':
        # import data set
        df = pd.read_csv(data_set_path, sep=';', parse_dates=True, index_col=0)
        # replace the nan values with 0
        df = df.replace(np.nan, 0)

        data = df.values

        # using keras often requires the data type float32
        data = data.astype('float32')

        data_length = len(data)

        # slice the data
        train_train = data[0:(data_length*80//100), :]           # 80%
        train_test = data[(data_length*80//100):data_length, :]  # 20%
        train = data[:, :]                                       # 100%

        if data is not None:
            model_is_ready_to_train[0] = True


def isitnumber(usr_input):
    return usr_input.replace('.', '', 1).isdigit()


def prepare_data(data, inputs_indices, outputs_indices):
    """
    Create lagged data from an input time series
    """
    global last_input_indices, last_output_indices
    x, y = [], []
    cntr = len(data) - outputs_indices[-1]
    for x_row in range(cntr):
        a, b = [], []
        for inputs_nmbr_row in inputs_indices:
            a.append(data[inputs_nmbr_row-1, 0])
        x.append(a)

        for outputs_nmbr_row in outputs_indices:
            b.append(data[outputs_nmbr_row-1, 0])
        y.append(b)

        inputs_indices = [t + 1 for t in inputs_indices]
        outputs_indices = [t+1 for t in outputs_indices]

    last_input_indices = [t-2 for t in inputs_indices]
    last_output_indices = [t-2 for t in outputs_indices]
    return np.array(x), np.array(y)


def acl_func(acl_data_x, lags):
    acldata = acl_data_x.flatten()
    mean = np.mean(acldata)
    numerator = 0
    denominator = 0
    for i in range(0, len(acldata)-lags):
        numerator += (acldata[i] - mean)*(acldata[i+lags] - mean)
    for i in range(0, len(acldata)-1):
        denominator += (acldata[i] - mean)**2
    return numerator/denominator


def check_train_fields():
    """ check if the the train fields is ready """

    if [i for i in model_is_ready_to_train if i is False]:
        messagebox.showinfo('Error: Missing Value', 'All field must be set. Please check it  and try again ')
    else:
        try:
            train_function()
        except :
            messagebox.showinfo('Logical Error',
                              'please choose a suitable inputs_nmbr values for your dataset and try again.')


def train_function():
    global model_is_ready_to_predict, mdl, UI, x_train

    # prepare the data
    x_train, y_train = prepare_data(train_train, inputs_indices, outputs_indices)
    x_test, y_test = prepare_data(train_test, inputs_indices, outputs_indices)

    UI.progress["maximum"] = 100
    best_layer_number = -1
    best_test_score = 9999999

    counter = min_neuron  # initialize counter
    progress_value = 0

    while counter <= (max_neuron + 1):
        # progress bar control
        progress_value += 100 / (2 * ((max_neuron - min_neuron) + 2))
        UI.progress["value"] = progress_value
        UI.update()

        # create the model
        mdl = Sequential()
        # prepare for the main model
        if counter == (max_neuron + 1):
            x_train, y_train = prepare_data(train, inputs_indices, outputs_indices)
            mdl.add(Dense(best_layer_number, input_dim=inputs_nmbr, activation='relu'))
            mdl.add(Dense(best_layer_number*2))
        else:
            mdl.add(Dense(counter, input_dim=inputs_nmbr, activation='relu'))
            mdl.add(Dense(counter*2))
        mdl.add(Dense(outputs_nmbr))
        mdl.compile(loss='mean_squared_error', optimizer='adam')

        # start the training
        mdl.fit(x_train, y_train, epochs=150, batch_size=1, verbose=2)

        # estimate model performance
        train_score = mdl.evaluate(x_train, y_train, verbose=0)
        test_score = mdl.evaluate(x_test, y_test, verbose=0)

        # progress bar control
        progress_value += 100 / (2 * ((max_neuron - min_neuron) + 2))
        UI.progress["value"] = progress_value
        UI.update()

        # find the minimum Test Score with RMSE unit
        if best_test_score > test_score and counter < max_neuron:
            best_test_score = test_score
            best_layer_number = counter
        elif counter is max_neuron:
            # save the model
            mdl.save('best_model.h5')

        counter += 1

    model_is_ready_to_predict[0] = True

    # prepare the model summary string
    summary = []
    mdl.summary(print_fn=lambda x: summary.append(x))
    short_model_summary = '\n'.join(summary)
    mdl.summary()

    # print the model information
    UI.results_dump.insert(INSERT, "The Best Neuron number is: {}\n".format(best_layer_number))
    UI.results_dump.insert(INSERT, 'The Best Test Score: {:.2f} MSE ({:.2f} RMSE)'
                           .format(test_score, math.sqrt(test_score)) + "\n")
    UI.results_dump.insert(INSERT, "Model Summary: \n{}".format(short_model_summary) + "\n")


def check_predict_fields():
    """ check if the model is finished the train and the predict fields is ready """

    if [i for i in model_is_ready_to_predict if i is False]:
        messagebox.showinfo('Error: Missing Value', 'All field must be set. Please check it  and try again ')
    else:
        try:
            predict_function()
        except :
            messagebox.showinfo('Logical Error',
                                'please choose a suitable values for the prediction model')


def predict_function():
    global first_values
    predict_array, predicted_array, f = [], [], []
    predict_array.append(first_values)
    for i in range(step_number):
        predicted_array = mdl.predict(np.array(predict_array))
        UI.results_dump.insert(INSERT, str(i) + " :  " + str(predicted_array[-1]) + "\n")

        if inputs_nmbr > outputs_nmbr:
            tmp = []
            for c in range(inputs_nmbr-outputs_nmbr, 0, -1):
                tmp.append(predict_array[-1][-1*c])
            for d in predicted_array[-1]:
                tmp.append(d)
        elif inputs_nmbr < outputs_nmbr:
            tmp = []
            for c in range(inputs_nmbr, 0, -1):
                tmp.append(predicted_array[-1][-1*c])
        else:
            tmp = predicted_array[-1]
        predict_array.append(tmp)
    np.savetxt("predicted_values_decimal_notation.csv", predicted_array, fmt='%f', delimiter=",")
    np.savetxt("predicted_values_scientific_notation.csv", predicted_array, delimiter=",")


def main():
    global UI
    UI = AppUI()
    UI.option_add('*font', ('verdana', 12, 'bold'))
    UI.title("N-step Predictor")
    UI.mainloop()


if __name__ == '__main__':
    main()
