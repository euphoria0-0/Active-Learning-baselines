data = 'CIFAR10'
filenames = {
    'Random' : 'Random-CIFAR10-2021-04-27-01-42-21.txt',
    'LearningLoss' : 'Learning-Loss-CIFAR10-2021-04-27-14-59-18.txt',
    #'VAAL' : 'VAAL-CIFAR10-2021-04-22-16-16-51.txt',
    'CoreSet' : 'CoreSet-CIFAR10-2021-04-27-03-54-19.txt',
    'BADGE' : 'BADGE-CIFAR10-2021-04-27-11-46-19.txt'
}


import seaborn as sns
import pandas as pd

'''for model in filenames.keys():
    filename = '../results/{}/{}/{}'.format(model,data,filenames[model])
    data = pd.read_csv(filename, skiprows=9)
    print(data.head())'''

data = pd.read_csv('C:/Users/unist-3/Documents/OneDrive - unist.ac.kr/UNIST/Research/AL/AL_results/CIFAR10.csv')
print(data.head())

sns.lineplot(
    data=data,
    x="CYCLE", y="Random",
    markers=True, dashes=False
)