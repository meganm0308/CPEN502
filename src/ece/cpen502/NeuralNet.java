package ece.cpen502;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
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
        double[][] deltaWHiddenToOutput = new double[numHiddenNeurons + 1][numOutputs];
        double[][] deltaWInputToHidden = new double[inputVectors.length][numHiddenNeurons + 1];

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
                deltaWHiddenToOutput[i][j] = momentum * deltaWHiddenToOutput[i][j]
                                            + learningRate * outputErrorSignals[j] * outputsHidden[i];
                hiddenToOutputWeights[i][j] += deltaWHiddenToOutput[i][j];
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
                hiddenErrorSignals[i] *= (1 - outputsHidden[i] * outputsHidden[i]) / 2.0;
            }
        }

        //update weights from the inputs to the hidden layers
        for (int i = 0; i < inputToHiddenWeights.length; i++) {
            for (int j = 0; j < inputToHiddenWeights[i].length; j++) {
                deltaWInputToHidden[i][j] = momentum * deltaWInputToHidden[i][j]
                                            + learningRate * hiddenErrorSignals[j] * inputVectors[currentTrainingSet][i];
                inputToHiddenWeights[i][j] += deltaWInputToHidden[i][j];
            }
        }
    }

    public ArrayList testError() {
        double[] outputsHidden;
        double[] outputs;
        double error;
        int epoch = 0;
        ArrayList<Double> errorList = new ArrayList<>();

        initializeWeights();

        do {
            currentTrainingSet = 0;
            error = 0;

            while (currentTrainingSet < inputVectors.length) {
                outputsHidden = forwardToHidden();
                outputs = forwardToOutput(outputsHidden);
                backPropagation(outputs, outputsHidden);

                for (int i = 0; i < outputs.length; i++) {
                    error += Math.pow((outputs[i] - expectedOutput[currentTrainingSet][i]),2);
                }
                currentTrainingSet++;
            }
            error = error / 2;
            epoch++;
            errorList.add(error);
        } while (error > errorThreshold);
        System.out.println("number of epochs: " + epoch);
        return errorList;
    }

    public static void textWriter(String fileName, ArrayList<Double> list) throws IOException {
        FileWriter writer = null;
        try {
            writer = new FileWriter(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (Double aDouble : list) {
            assert writer != null;
            writer.write(aDouble + ",");
        }
        assert writer != null;
        writer.close();
    }

    public static void main(String[] args) throws IOException {

        double learningRate = 0.2;
        int noOfHiddenNeurons = 4;

        //two different inputs
        double[][] binaryInput = {{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
        double[][] binaryExpectedOutput = {{0},{1},{1},{0}};

        double[][] bipolarInput = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};
        double[][] bipolarExpectedOutput = {{-1}, {1}, {1}, {-1}};

        NeuralNet BinaryNoMomentum = new NeuralNet(binaryInput, binaryExpectedOutput, learningRate, 0, noOfHiddenNeurons, true);
        NeuralNet BipolarNoMomentum = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate,0, noOfHiddenNeurons, false);
        NeuralNet BipolarWithMomentum = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate,0.9, noOfHiddenNeurons, false);

        ArrayList xorErrors = BinaryNoMomentum.testError();
        textWriter("BinaryNoMomentum.txt", xorErrors);

        ArrayList bipolarErrors = BipolarNoMomentum.testError();
        textWriter("BipolarNoMomentum.txt", bipolarErrors);

        ArrayList bipolarMomentumErrors = BipolarWithMomentum.testError();
        textWriter("BipolarWithMomentum.txt", bipolarMomentumErrors);
    }
}
