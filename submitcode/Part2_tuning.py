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
        #Just for debugging, makes sure code is actually proceeding
        #if(i%100==0):
        #    print("Has run uptill i=",i);
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

    klist= np.array((1,5,8,12,15,19,25));
    #klist= np.array((15,16,17,18,19,20,21,22,23,24,25,27,30));
    kaccuracies=np.zeros(7);
    # Get training  file name from the command line
    traindatafile = sys.argv[1]
    

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray();
    Ytr = tr_data[1];
    T=40000; #To pick cross validation set later
    Ytr = Ytr.reshape(Ytr.shape[0],1);
    #Concatenating Xtr and Ytr to shuffle them together
    Ztr = np.concatenate((Xtr,Ytr),axis=1);
    np.random.shuffle(Ztr); # inplace shuffled
    temp=Ztr[:T]; # first T data points go to train
    Xtr = temp[:,:-1]; # getting the X  part out
    Ytr = temp[:,-1]; # y label part out
    temp = Ztr[T:]

    #only the names are Xts and trueYts, test data has not been touched
    Xts = temp[:,:-1] # getting the X  part out
    trueYts = temp[:,-1]; # y label part out
    #trueYts = trueYts.reshape(trueYts.shape[0],1);
   
    Ntest= Xts.shape[0];
    c=-1;
    for k in klist:
        #print("For k = ",k);
        c+=1;
        Yts = predict(Xtr, Ytr, Xts,k);
        ansBool = Yts==trueYts;
        ans=ansBool.sum(0);   # gets the number of correct predictions
        accuracy = 100*(ans/Ntest);
        #print("Accuracy is ",accuracy);
        kaccuracies[c]=accuracy;

    #Plotted the data for my own convenience
    plt.plot(klist,kaccuracies,linestyle='dashed', marker="o", color="green");
    #plt.xticks([1,2,3,5,10]);
    plt.xlabel("K Values ->");
    plt.ylabel("Percentage Accuracies(%)");
    plt.title("Euclidean K-NN Cross Validation Analysis");
    plt.show();

if __name__ == '__main__':
    main()
