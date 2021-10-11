package ece.cpen502;

import java.util.Arrays;
import java.util.Random;

public class NeuralNet {

    private boolean isBinary; //true = binary training sets used, false = bipolar training sets
    private double learningRate, momentum;
    private int numHiddenNeurons;

    //hyper-parameters
    private static double errorThreshold = 0.05;
    private static int numInputs = 2;
    private static int numOutputs = 1;
    private static int currentTrainingSet = 0;

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

        inputToHiddenWeights = new double[numInputs + 1][numHiddenNeurons+1];
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
        double[] outputsHidden = new double[numHiddenNeurons + 1];
        outputsHidden[0] = biasInput;

        //outputs from the hidden neurons
        for (int i = 1; i < outputsHidden.length; i++) {
            outputsHidden[i] = 0;
            for (int j = 0; j < inputToHiddenWeights.length; j++) {
                outputsHidden[i] += inputVectors[currentTrainingSet][j] * inputToHiddenWeights[j][i];
            }
            outputsHidden[i] = sigmoid(outputsHidden[i]);  //apply activation function
        }
        return outputsHidden;
    }

    public double[] forwardToOutput(double[] outputsHidden) {
        double[] outputs = new double[numOutputs];
        //outputs from the output neuron
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = 0;
            for (int j = 0; j < hiddenToOutputWeights.length; j++) {
                outputs[i] += outputsHidden[j] * hiddenToOutputWeights[j][i];
            }
            outputs[i] = sigmoid(outputs[i]); //apply activation function

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


                hiddenToOutputWeights[i][j] += learningRate * outputErrorSignals[j] * outputsHidden[i];

            }
        }

        //compute the error signals at the hidden neurons
        for (int i = 0; i < hiddenErrorSignals.length; i++) {
            hiddenErrorSignals[i] = 0;
            for (int j = 0; j < numOutputs; j++) {
                hiddenErrorSignals[i] += hiddenToOutputWeights[i][j] * outputErrorSignals[j];
            }
            if (isBinary) {


                hiddenErrorSignals[i] *= outputsHidden[i] * (1 - outputsHidden[i]);

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
                inputToHiddenWeights[i][j] += learningRate * hiddenErrorSignals[j] * inputVectors[currentTrainingSet][i];
            }
        }
    }

    public void testError() {
        double[] outputsHidden;
        double[] outputs;
        double error;
        int epoch = 0;

        initializeWeights();

        do {
            currentTrainingSet = 0;
            error = 0;

            while (currentTrainingSet < inputVectors.length) {
                outputsHidden = forwardToHidden();
                outputs = forwardToOutput(outputsHidden);
                for (int i = 0; i < outputs.length; i++) {
                    error += Math.pow((outputs[i] - expectedOutput[currentTrainingSet][i]),2);
                }

                if (currentTrainingSet == (inputVectors.length - 1)) {
                    error = error / 2;
                    backPropagation(outputs, outputsHidden);
                    epoch++;
                    if (epoch == 1) {
                        System.out.println(error);
                    }
                    if (epoch == 200) {
                        System.out.println(error);
                    }
                }
                currentTrainingSet++;
            }
        } while (error > errorThreshold);
        System.out.println("number of epochs: " + epoch);
    }

    public static void main(String[] args) {

        double learningRate = 0.2;
        int noOfHiddenNeurons = 4;

        //two different inputs
        double[][] binaryInput = {{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
        double[][] binaryExpectedOutput = {{0},{1},{1},{0}};

        double[][] bipolarInput = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};
        double[][] bipolarExpectedOutput = {{-1}, {1}, {1}, {-1}};

        NeuralNet XOR1 = new NeuralNet(binaryInput, binaryExpectedOutput, learningRate, 0, noOfHiddenNeurons, true);
        NeuralNet Bipolar = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate,0, noOfHiddenNeurons, false);
        XOR1.testError();
        //Bipolar.testError();


//        double[] outputsHidden;
//        double[] outputs;
//        double error = 0;
//        XOR1.initializeWeights();
//
//        outputsHidden = XOR1.forwardToHidden();
//        outputs = XOR1.forwardToOutput(outputsHidden);
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        currentTrainingSet++;
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        currentTrainingSet++;
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        currentTrainingSet++;
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        error = error/2;
//        System.out.println(error);
//
//        System.out.println(Arrays.deepToString(XOR1.inputToHiddenWeights));
//        System.out.println(Arrays.deepToString(XOR1.hiddenToOutputWeights));
//
//        XOR1.backPropagation(outputs, outputsHidden);
//        System.out.println(Arrays.deepToString(XOR1.inputToHiddenWeights));
//        System.out.println(Arrays.deepToString(XOR1.hiddenToOutputWeights));
//
//        currentTrainingSet = 0;
//        error = 0;
//        outputsHidden = XOR1.forwardToHidden();
//        outputs = XOR1.forwardToOutput(outputsHidden);
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        currentTrainingSet++;
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        currentTrainingSet++;
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        currentTrainingSet++;
//        for (int i = 0; i < outputs.length; i++) {
//            error += Math.pow((outputs[i] - binaryExpectedOutput[currentTrainingSet][i]),2);
//        }
//        System.out.println(error);
//        System.out.println(currentTrainingSet);
//
//        error = error/2;
//        System.out.println(error);
    }
}
