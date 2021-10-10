package ece.cpen502;

import java.util.Arrays;
import java.util.Random;

public class NeuralNet {

    private boolean isBinary; //true = binary training sets used, false = bipolar training sets
    private double learningRate, momentum;
    private int numHiddenNeurons;

    //hyper-parameters
    public static double errorThreshold = 0.05;
    private static int numInputs = 2;
    private static int numOutputs = 1;
    private int currentTrainingSet = 0;

    //upper and lower bounds for initializing weights
    private double weightMin = -0.5;
    private double weightMax = 0.5;

    //weights
    private double[][] inputToHiddenWeights, hiddenToOutputWeights; //+1 to accommodate a bias weight

    //inputs
    private double biasInput = 1;
    private double[][] inputVectors, expectedOutput;

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

    //Forward propagation to calculate the outputs from the hidden neurons and the output neuron(s)
    public double[] forwardToHidden() {
        double[] outputsHidden = new double[numHiddenNeurons];

        //outputs from the hidden neurons
        for (int i = 0; i < outputsHidden.length; i++) {
            outputsHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                outputsHidden[i] += inputVectors[currentTrainingSet][j] * inputToHiddenWeights[j][i];
                outputsHidden[i] = sigmoid(outputsHidden[i]);  //apply activation function
            }
        }
        return outputsHidden;
    }

    public double[] forwardToOutput(double[] outputsHidden) {
        double[] outputs = new double[numOutputs];
        //outputs from the output neuron
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = 0;
            for (int j = 0; j < hiddenToOutputWeights.length; j++) {
                if (j==0) {  //first weight applied to the bias input
                    outputs[i] += biasInput * hiddenToOutputWeights[j][i];
                } else {
                    outputs[i] += outputsHidden[j-1] * hiddenToOutputWeights[j][i];
                }
                outputs[i] = sigmoid(outputs[i]); //apply activation function
            }
        }
        return outputs;
    }

    public void backPropagation(double[] outputs, double[] outputsHidden) {

        double[] outputErrorSignals = new double[numOutputs];
        double[] hiddenErrorSignals = new double[numHiddenNeurons + 1];

        //compute the error signals at the outputs neurons
        if (isBinary) {
            for (int i = 0; i < outputs.length; i++) {
                outputErrorSignals[i] = (expectedOutput[currentTrainingSet][i] - outputs[i]) *
                                        outputs[i] * (1 - outputs[i]);
            }
        } else {
            for (int i = 0; i < outputs.length; i++) {
                outputErrorSignals[i] = (expectedOutput[currentTrainingSet][i] - outputs[i]) *
                                        (1 - outputs[i] * outputs[i]) / 2.0;
            }
        }

        //update weights from the hidden layer to the outputs
        for (int i = 0; i < hiddenToOutputWeights.length; i++) {
            for (int j = 0; j < hiddenToOutputWeights[i].length; j++) {
                if (i == 0) {  //bias input at the hidden layer
                    hiddenToOutputWeights[i][j] += learningRate * outputErrorSignals[j] * biasInput;
                } else {
                    hiddenToOutputWeights[i][j] += learningRate * outputErrorSignals[j] * outputsHidden[i-1];
                }
            }
        }

        //compute the error signals at the hidden neurons
        for (int i = 0; i < hiddenErrorSignals.length; i++) {
            for (int j = 0; j < numOutputs; j++) {
                hiddenErrorSignals[i] += hiddenToOutputWeights[i][j] * outputErrorSignals[j];
            }
            if (isBinary) {
                if (i == 0) {
                    hiddenErrorSignals[i] *= biasInput * (1 - biasInput);
                } else {
                    hiddenErrorSignals[i] *= outputsHidden[i-1] * (1 - outputsHidden[i-1]);
                }
            } else {
                if (i == 0) {
                    hiddenErrorSignals[i] *= (1 - biasInput * biasInput) / 2.0;
                } else {
                    hiddenErrorSignals[i] *= (1 - outputsHidden[i-1] * outputsHidden[i-1]) / 2.0;
                }
            }
        }

        //update weights from the inputs to the hidden layers
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 0; j < inputToHiddenWeights[i].length; j++) {
                inputToHiddenWeights[i][j] += learningRate * hiddenErrorSignals[j+1] * inputVectors[currentTrainingSet][i];
            }
        }
    }

    public static void main(String[] args) {

        double momentum = 0;
        double learningRate = 0.2;
        int noOfHiddenNeurons = 4;
        double[] outputsHidden;
        double[] outputs;

        //two different inputs
        double[][] binaryInput = {{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
        double[][] binaryExpectedOutput = {{0},{1},{1},{0}};

        double[][] bipolarInput = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};
        double[][] bipolarExpectedOutput = {{-1}, {1}, {1}, {-1}};

        NeuralNet XOR = new NeuralNet(binaryInput, binaryExpectedOutput, learningRate, momentum, noOfHiddenNeurons, true);
        XOR.initializeWeights();
        //do {
        System.out.println(Arrays.deepToString(XOR.inputToHiddenWeights));
            outputsHidden = XOR.forwardToHidden();
        System.out.println(Arrays.toString(outputsHidden));
            outputs = XOR.forwardToOutput(outputsHidden);
        System.out.println(Arrays.toString(outputs));
            XOR.backPropagation(outputs,outputsHidden);
        System.out.println(Arrays.deepToString(XOR.inputToHiddenWeights));

        //}
    }
}
