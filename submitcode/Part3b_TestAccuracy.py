from sklearn.datasets import load_svmlight_file
import numpy as np
import sys
import matplotlib.pyplot as plt;
def predict(Xtr, Ytr, Xts,k, metric=None):

    N, D = Xtr.shape

    '''number of rows in Ytr'''
    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    if metric is None:
        metric = np.identity(D)

    Yts = np.zeros(Xts.shape[0])
    for i in range(Xts.shape[0]):
        #if(i%100==0):
        #    print("Running for i = ",i);
        xtest = Xts[i,:];
        Dist = xtest - Xtr;  #Numpy broadcasts smaller array
        DistTranspose = Dist.transpose();
        NewDistTranspose = np.matmul(metric,DistTranspose);   # The step where the Mahalanobis metric is multiplied 
        NewDist = NewDistTranspose.transpose();
        temp = Dist*NewDist;
        squaredDistArr = temp.sum(1);      #Is a N x 1 matrix with square of distances of the test point with all training points
        yInd = np.argsort(squaredDistArr);
        sortedLabel = Ytr[yInd];         
        knn = sortedLabel[:k];
        values, counts = np.unique(knn, return_counts=True);
        Yts[i] = values[np.argmax(counts)];

    return Yts

def main():
    k= 5;  #Chosen during tuning step in Part2

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

    # Load the learned metric
    metric = np.load("model.npy")
   # print("The mtric first row: ",metric[:,0]);
   # print("Shape of metric",metric.shape);
    Ntest= Xts.shape[0];
    #print("shapes od Xtr,Ytr,Xts,trueYts",Xtr.shape,Ytr.shape,Xts.shape,trueYts.shape);
    #print("For k = ",k);
    Yts = predict(Xtr, Ytr, Xts,k,metric);
    ansBool = Yts==trueYts;
    ans=ansBool.sum(0);
    accuracy = 100*(ans/Ntest);
    #print("Accuracy is ",accuracy);
    



    # Save predictions to a file, just for my own reference
    np.savetxt("testYk5.dat",Yts)


if __name__ == '__main__':
    main()
