package ece.cpen502;

import java.util.Arrays;
import java.util.Random;

public class NeuralNet {

    public static boolean isBinary = true; //true = binary training sets used, false = bipolar training sets

    //hyper-parameters
    public static double learningRate = 0.2;
    public static double momentum = 0;
    public static double errorThreshold = 0.05;

    int numInputs = 2;
    int numHiddenNeurons = 4;
    int numOutputs = 1;
    int currentTrainingSet = 0;

    //upper and lower bounds for initializing weights
    double weightMin = -0.5;
    double weightMax = 0.5;

    //weights
    double[][] inputToHiddenWeights = new double[numInputs + 1][numHiddenNeurons]; //+1 to accommodate a bias weight
    double[][] hiddenToOutputWeights = new double[numHiddenNeurons + 1][numOutputs]; //+1 to accommodate a bias weight

    //inputs
    double biasInput = 1;
    double[][] binarySets = {{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
    double[][] bipolarSets = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};

    //outputs
    double[] outputsHidden = new double[numHiddenNeurons];
    double[] output = new double[numOutputs];

    //error
    double[] errorSets = new double[numOutputs];

    //Initialize weights to random values in the range [weightMin, weightMax]
    public void initializeWeights() {

        //Initialize weights from the inputs to the neurons at the hidden layer
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 0; j < inputToHiddenWeights[i].length; j++) {
                inputToHiddenWeights[i][j] = weightMin + (new Random().nextDouble() * (weightMax - weightMin));
            }
        }

        for (int i = 0; i < hiddenToOutputWeights.length; i++) {
            for (int j = 0; j < hiddenToOutputWeights[i].length; j++) {
                hiddenToOutputWeights[i][j] = weightMin + (new Random().nextDouble() * (weightMax - weightMin));
            }
        }
    }

    //The activation function
    public double sigmoid(double x) {
        if (isBinary) {
            return 1 / (1 + Math.pow(Math.E, -x)); //sigmoid function for binary training sets
        }
        else {
            return -1 + 2 / (1 + Math.pow(Math.E, -x)); //sigmoid function for bipolar training sets
        }
    }

    //Forward propagation to calculate the output from the hidden neurons and the output neuron
    public void forwardPropagation() {

        //outputs from the hidden neurons
        for (int i = 0; i < outputsHidden.length; i++) {
            outputsHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                outputsHidden[i] += binarySets[currentTrainingSet][j] * inputToHiddenWeights[j][i];
                outputsHidden[i] = sigmoid(outputsHidden[i]);
            }
        }

        //outputs from the output neuron
        for (int i = 0; i < output.length; i++) {
            output[i] = 0;
            for (int j = 0; j < hiddenToOutputWeights.length; j++) {
                if (j==0) {  //first weight applied to the bias input
                    output[i] += biasInput * hiddenToOutputWeights[j][i];
                } else {
                    output[i] += outputsHidden[j-1] * hiddenToOutputWeights[j][i];
                }
                output[i] = sigmoid(output[i]);
            }
        }
    }



    public static void main(String[] args) {
        NeuralNet XOR = new NeuralNet();
        //XOR.initializeWeights();
        //XOR.forwardPropagation();
    }
}
