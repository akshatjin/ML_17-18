from sklearn.datasets import load_svmlight_file
import numpy as np
import sys
import matplotlib.pyplot as plt;
def predict(Xtr, Ytr, Xts,k, metric=None):

    N, D = Xtr.shape

    
    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    if metric is None:
        metric = np.identity(D)

    Yts = np.zeros(Xts.shape[0])
    for i in range(Xts.shape[0]):
        xtest = Xts[i,:];
        Dist = xtest - Xtr;  #We use the broadcasting capability of numpy here
        temp = Dist*Dist;    
        squaredDistArr = temp.sum(1);  # Row wise sum gives a N x 1 array storing squares of distances of the test point with all training points
        yInd = np.argsort(squaredDistArr);
        sortedLabel = Ytr[yInd];        
        knn = sortedLabel[:k];
        values, counts = np.unique(knn, return_counts=True);   # helps in counting the number of occurences of each label
        Yts[i] = values[np.argmax(counts)];

    return Yts

def main(): 
    klist= np.array((1,2,3,5,10));
    kaccuracies=np.zeros(5);
    # Get training and testing file names from the command line
    traindatafile = sys.argv[1]
    testdatafile = sys.argv[2]

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray();
    Ytr = tr_data[1];

    # The testing file is in libSVM format too
    ts_data = load_svmlight_file(testdatafile)

    Xts = ts_data[0].toarray();
    trueYts = ts_data[1];

  
    Ntest= Xts.shape[0];
    c=-1;
    for k in klist:
        #print("For k = ",k);
        c+=1;
        Yts = predict(Xtr, Ytr, Xts,k);
        ansBool = Yts==trueYts;
        ans=ansBool.sum(0);
        accuracy = 100*(ans/Ntest);
        #print("Accuracy is ",accuracy);
        kaccuracies[c]=accuracy;

    #Printing the plot
    plt.plot(klist,kaccuracies,linestyle='dashed', marker="o", color="green");
    plt.xticks([1,2,3,4,5,6,7,8,9,10]);
    plt.xlabel("K Values ->");
    plt.ylabel("Percentage Accuracies(%)");
    plt.title("Euclidean K-NN Test Analysis");
    plt.show();

if __name__ == '__main__':
    main()
