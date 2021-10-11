package ece.cpen502;

import org.junit.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Assert;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NeuralNetTester {
    NeuralNet testBinaryNN;
    NeuralNet testBipolarNN;
    double[][] inputToHiddenWeights ={{0.1, 0.1, 0.1}, {0.2, 0.2, 0.2}, {-0.1, -0.1, -0.1}, {-0.2, -0.2, -0.2}};

    double[][] hiddenToOutputWeights;

    @BeforeEach
    void setup(){
        double momentum = 0;
        double learningRate = 0.2;
        int noOfHiddenNeurons = 4;

        //two different inputs
        double[][] binaryInput = {{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
        double[][] binaryExpectedOutput = {{0},{1},{1},{0}};

        double[][] bipolarInput = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};
        double[][] bipolarExpectedOutput = {{-1}, {1}, {1}, {-1}};
        testBinaryNN = new NeuralNet(binaryInput, binaryExpectedOutput, learningRate, momentum, noOfHiddenNeurons, true);
        testBipolarNN = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate,momentum, noOfHiddenNeurons, false);

//        inputToHiddenWeights = new double[][];
        hiddenToOutputWeights = new double[5][1];
    }

    @Test
    public void testSigmoid() {
//        NeuralNet nnTest = new NeuralNet();
        // TODO
        double x= 0.0;
        assertEquals(20, testBinaryNN.sigmoid(x), "sigmoid calculation should be correct");
    }

    public void testForwardHidden() {
//        NeuralNet nnTest = new NeuralNet();
        // TODO

        testBinaryNN.setInputToHiddenWeights(inputToHiddenWeights);
        assertEquals(20, testBinaryNN.sigmoid(x), "sigmoid calculation should be correct");
    }
}
