package ece.cpen502;

import java.util.Arrays;
import java.util.Random;

public class NeuralNet {

    private boolean isBinary; //true = binary training sets used, false = bipolar training sets
    private double learningRate, momentum;
    private int numHiddenNeurons;

    //hyper-parameters
    public static double errorThreshold = 0.05;

    private int numInputs = 2;
    private int numOutputs = 1;
    private int currentTrainingSet = 0;

    //upper and lower bounds for initializing weights
    private double weightMin = -0.5;
    private double weightMax = 0.5;

    //weights
    private double[][] inputToHiddenWeights, hiddenToOutputWeights; //+1 to accommodate a bias weight

    //inputs
    private double biasInput = 1;
    private double[][] inputVectors, expectedOutput;

    //outputs
    private double[] outputsHidden = new double[numHiddenNeurons];
    private double[] output = new double[numOutputs];
    //error
    private double[] errorSets = new double[numOutputs];

    NeuralNet (double[][] input, double[][] output,
                double lrnRate, double inputMomentum,
                int noOfHiddenNeurons, boolean isBinaryTraining) {
        inputVectors = input;
        expectedOutput = output;
        learningRate = lrnRate;
        momentum = inputMomentum;
        numHiddenNeurons = noOfHiddenNeurons;
        isBinary = isBinaryTraining;

        inputToHiddenWeights = new double[numInputs + 1][numHiddenNeurons];
        hiddenToOutputWeights = new double[numHiddenNeurons + 1][numOutputs];
    }

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
        } else {
            return -1 + 2 / (1 + Math.pow(Math.E, -x)); //sigmoid function for bipolar training sets
        }
    }

    //Forward propagation to calculate the output from the hidden neurons and the output neuron
    public void forwardPropagation() {

        //outputs from the hidden neurons
        for (int i = 0; i < outputsHidden.length; i++) {
            outputsHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                outputsHidden[i] += inputVectors[currentTrainingSet][j] * inputToHiddenWeights[j][i];
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

        double momentum = 0;
        double learningRate = 0.2;
        int noOfHiddenNeurons = 4;

        //two different inputs
        double[][] binaryInput = {{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
        double[][] expectedOutput = {{}};

        double[][] bipolarInput = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};

        NeuralNet XOR = new NeuralNet(binaryInput, expectedOutput, learningRate, momentum, noOfHiddenNeurons, true);
        XOR.initializeWeights();
        //XOR.forwardPropagation();
    }
}
