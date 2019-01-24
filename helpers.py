from sklearn.model_selection import validation_curve, learning_curve
import numpy as np
import matplotlib.pyplot as plt

def make_complexity_curve(clf, x, y,param_name,param_range,cv,clf_name,dataset_name):
    title = 'Model Complexity Curve: {} - {} ({})'.format(clf_name, dataset_name, param_name)
    print(title)
    train_scores, validation_scores = validation_curve(clf,x,y,param_name,param_range,cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    
    best = max(list(zip(param_range, validation_scores_mean)), key = lambda t: t[1])[0]
    print(f"best {param_name}: {best}")
    
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("accuracy")
    plt.plot(param_range,train_scores_mean, label="train")
    plt.plot(param_range, validation_scores_mean, label="validation")
    plt.legend()
    plt.plot()
    plt.show()
    print("Complexity curve complete")


def make_learning_curve(clf, x, y,train_sizes,cv,clf_name, dataset_name):
    title = 'Learning Curve: {} - {}'.format(clf_name, dataset_name)
    train_sizes, train_scores, validation_scores = learning_curve(clf, x, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    train_sizesi = [i for i in range(0, len(train_sizes))]
    
    best = max(list(zip(train_sizes, validation_scores_mean)), key = lambda t: t[1])[0]
    print(f"best size: {best}")
    plt.close()
    plt.figure()
    plt.title(title)
    plt.xticks(train_sizesi, train_sizes)
    plt.xlabel("number of samples")
    plt.ylabel("accuracy")
    plt.plot(train_sizes,train_scores_mean, label="train")
    plt.plot(train_sizes, validation_scores_mean, label="validation")
    plt.legend()
    plt.draw()