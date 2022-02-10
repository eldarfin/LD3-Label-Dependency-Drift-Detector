import matplotlib as mpl
mpl.rcParams['figure.dpi'] =600
accs_= []
hamms_ = []
f1exs_ = []
f1mics_ = []
f1macs_ = []

for i in [0,2,3,4,1]:
    index = len(results_pred[i]) // 25
    accs= []
    hamms = []
    f1exs = []
    f1mics = []
    f1macs = []
    for j in range(25):
        accs.append(np.round(util.accuracy_example(np.array([results_true[j*index:(j+1)*index]]), np.array([results_pred[i][j*index:(j+1)*index]])), decimals=4))
        hamms.append(np.round(util.hamming_loss(np.array(results_true[j*index:(j+1)*index]), np.array(results_pred[i][j*index:(j+1)*index])), decimals=4))
        f1exs.append(np.round(f1_score(np.array(results_true[j*index:(j+1)*index]), np.array(results_pred[i][j*index:(j+1)*index]), average='samples'),decimals=4))
        f1mics.append(np.round(util.f1_micro(np.array(results_true[j*index:(j+1)*index]), np.array(results_pred[i][j*index:(j+1)*index])), decimals=4))
        f1macs.append(np.round(util.f1_macro(np.array(results_true[j*index:(j+1)*index]), np.array(results_pred[i][j*index:(j+1)*index])), decimals=4))
    accs_.append(accs)
    hamms_.append(hamms)
    f1exs_.append(f1exs)
    f1mics_.append(f1mics)
    f1macs_.append(f1macs)









arange = np.arange(1, 101, step=4)
colors = ['mediumblue', 'palegreen', 'violet', 'orange', 'aquamarine']
markers = ['o', '^', 'x', 's', '*']
labels = ['LD3', 'ADWIN', 'EDDM', 'RDDM', 'ND']
for i in range(5):
    plt.plot(arange, accs_[i], color=colors[i], marker=markers[i], linestyle='-', label=labels[i])
    #plt.plot(arange, accs_[i], colors[i], label=labels[i])

plt.xlabel('Percentage of data')
plt.xticks([0,20,40,60,80,100])
plt.ylabel('Example-based accuracy')
plt.legend(loc='lower right')
plt.show()








arange = np.arange(1, 101, step=4)
colors = ['ro-', 'g^-', 'bx-', 'ms-', 'y*-']
labels = ['LD3', 'ADWIN', 'EDDM', 'RDDM', 'ND']
for i in range(5):
    plt.plot(arange, hamms_[i], colors[i], label=labels[i])

plt.xlabel('Percentage of data')
plt.xticks([0,20,40,60,80,100])
plt.ylabel('Hamming Loss')
plt.legend(loc='lower right')
plt.show()







arange = np.arange(1, 101, step=4)
colors = ['ro-', 'g^-', 'bx-', 'ms-', 'y*-']
labels = ['LD3', 'ADWIN', 'EDDM', 'RDDM', 'ND']
for i in range(5):
    plt.plot(arange, f1exs_[i], colors[i], label=labels[i])

plt.xlabel('Percentage of data')
plt.xticks([0,20,40,60,80,100])
plt.ylabel('Example-based F1 score')
plt.legend(loc='lower right')
plt.show()










arange = np.arange(1, 101, step=4)
colors = ['ro-', 'g^-', 'bx-', 'ms-', 'y*-']
labels = ['LD3', 'ADWIN', 'EDDM', 'RDDM', 'ND']
for i in range(5):
    plt.plot(arange, f1mics_[i], colors[i], label=labels[i])

plt.xlabel('Percentage of data')
plt.xticks([0,20,40,60,80,100])
plt.ylabel('Micro-averaged F1 score')
plt.legend(loc='lower right')
plt.show()














arange = np.arange(1, 101, step=4)
colors = ['ro-', 'g^-', 'bx-', 'ms-', 'y*-']
labels = ['LD3', 'ADWIN', 'EDDM', 'RDDM', 'ND']
for i in range(5):
    plt.plot(arange, f1macs_[i], colors[i], label=labels[i])

plt.xlabel('Percentage of data')
plt.xticks([0,20,40,60,80,100])
plt.ylabel('Macro-averaged F1 score')
plt.legend(loc='lower right')
plt.show()