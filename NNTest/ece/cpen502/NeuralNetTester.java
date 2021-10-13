package ece.cpen502;

import org.junit.Test;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Assert;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NeuralNetTester {
    double[][] inToHidWeights2 = {{0.1, 0.2, 0.3, 0.4}, {-0.1, -0.2, -0.3, -0.4}, {0.5, 0.6, 0.7, 0.8}};
    double[][] hidToOutWeights2 = {{0.1}, {0.2}, {-0.3}, {-0.4}, {0.5}};

    double[][] binaryInput2 = {{1,1,1}, {1,0,0}, {1,1,0}, {0,0,1}};
    double[][] binaryExpectedOutput2 = {{0},{0},{1},{1}};

    double[][] bipolarInput = {{1,-1,-1}, {1,-1,1}, {1,1,-1}, {1,1,1}};
    double[][] bipolarExpectedOutput = {{-1}, {1}, {1}, {-1}};

    double momentum = 0;
    double learningRate = 0.2;
    int noOfHiddenNeurons = 4;

    @Test
    public void testSigmoidForBinary() {
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        double x = 0.1;
        double expectedResult = 1/(1+Math.exp(-x));
        assertEquals(expectedResult, nnTest.sigmoid(x), "sigmoid calculation is incorrect");
    }

    @Test
    public void testSigmoidForBipolar() {
        NeuralNet nnTest = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate, momentum, noOfHiddenNeurons, false);
        double x = 0.1;
        double expectedResult = -1 + 2 / (1 + Math.pow(Math.E, -x));
        assertEquals(expectedResult, nnTest.sigmoid(x), "sigmoid calculation is incorrect");
    }

    @Test
    public void testForwardToHiddenForBinary() {
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        nnTest.setInputToHiddenWeights(inToHidWeights2);
        //test case is input {1,1}
        double s1 = 1*0.1 + 1*(-0.1)+1*0.5;
        double s2 = 1*0.2 + 1*(-0.2)+1*0.6; //0.6
        double s3 = 1*0.3 + 1*(-0.3)+1*(0.7); //0.7
        double s4 = 1*0.4 + 1*(-0.4)+1*(0.8); // 0.8
        double y1 = nnTest.sigmoid(s1);
        double y2 = nnTest.sigmoid(s2);
        double y3 = nnTest.sigmoid(s3);
        double y4 = nnTest.sigmoid(s4);
        double[] expectedOutputsHiddenNeurons = {1.0, y1, y2, y3, y4};
        Assert.assertArrayEquals(expectedOutputsHiddenNeurons, nnTest.forwardToHidden(), 0);
    }

    @Test
    public void testForwardToHiddenForBipolar() {
        double[][] testInput = {bipolarInput[0]};
        double[][] testOutput = {bipolarExpectedOutput[0]};
        NeuralNet nnTest = new NeuralNet(testInput, testOutput, learningRate, momentum, noOfHiddenNeurons, false);
        nnTest.setInputToHiddenWeights(inToHidWeights2);
        //test case is input {-1,-1}
        double s1 = 1*0.1 + -1*(-0.1) + -1*0.5;
        double s2 = 1*0.2 + -1*(-0.2) + -1*0.6;
        double s3 = 1*0.3 + -1*(-0.3) + -1*(0.7);
        double s4 = 1*0.4 + -1*(-0.4) + -1*(0.8);
        double y1 = nnTest.sigmoid(s1);
        double y2 = nnTest.sigmoid(s2);
        double y3 = nnTest.sigmoid(s3);
        double y4 = nnTest.sigmoid(s4);
        double[] expectedOutputsHiddenNeurons = {1, y1, y2, y3, y4};
//        System.out.println(Arrays.toString(expectedOutputsHiddenNeurons));
//        System.out.println(Arrays.toString(nnTest.forwardToHidden()));
        Assert.assertArrayEquals(expectedOutputsHiddenNeurons, nnTest.forwardToHidden(), 0);
    }

    @Test
    public void testForwardToOutputForBinary() {
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        nnTest.setHiddenToOutputWeights(hidToOutWeights2);
        double[] mockOutputsHidden = {1, 0.3, 0.4, 0.5, 0.6};
        double s = 1*0.1+ 0.3*0.2 + 0.4*(-0.3) + 0.5*(-0.4)+0.6*0.5;
        double y = nnTest.sigmoid(s);
        double[] expectedOutputs = {y};
        Assert.assertArrayEquals(expectedOutputs, nnTest.forwardToOutput(mockOutputsHidden), 0);
    }

    @Test
    public void testForwardToOutputForBipolar() {
        NeuralNet nnTest = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate, momentum, noOfHiddenNeurons, false);
        nnTest.setHiddenToOutputWeights(hidToOutWeights2);
        double[] mockOutputsHidden = {1, 0.3, 0.4, 0.5, 0.6};
        double s = 1*0.1+ 0.3*0.2 + 0.4*(-0.3) + 0.5*(-0.4)+0.6*0.5;
        double y = nnTest.sigmoid(s);
        double[] expectedOutputs = {y};
        Assert.assertArrayEquals(expectedOutputs, nnTest.forwardToOutput(mockOutputsHidden), 0);
    }

    @Test
    public void testBackPropagationUpdateInToHiddenForBinary() {
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        nnTest.setInputToHiddenWeights(inToHidWeights2);
        nnTest.setHiddenToOutputWeights(hidToOutWeights2);

        double[] mockOutputsHidden = {1, 0.3, 0.4, 0.5, 0.6};
        double[] mockOutputs = {0.5};
        double f_prime_y = mockOutputs[0]*(1-mockOutputs[0]);
        double sigma_y = f_prime_y*(binaryExpectedOutput2[0][0]-mockOutputs[0]);
        double[] f_prime_hidden = new double[5];
        double[] sigma_hidden = new double[5];
        double[][] expectedHiddenToOutWeights = new double[5][1];

        for (int i=0; i< 5; i++){
            double currentW = hidToOutWeights2[i][0];
            expectedHiddenToOutWeights[i][0] = currentW + learningRate* sigma_y*mockOutputsHidden[i];
        }

        for (int i=0; i< 5; i++){
            f_prime_hidden[i] = mockOutputsHidden[i]*(1-mockOutputsHidden[i]);
            double currentW = expectedHiddenToOutWeights[i][0];
            sigma_hidden[i] = currentW*sigma_y*f_prime_hidden[i];
        }

        double[][] expectedInputToHiddenUpdate = new double[3][4];
        for (int i1 = 0; i1 < 3; i1++) {
            for (int j = 0; j < 4; j++) {
                    double currentW = inToHidWeights2[i1][j];
                    double currentInput = binaryInput2[0][i1];
                    double currentSigma = sigma_hidden[j+1];
                expectedInputToHiddenUpdate[i1][j] = currentW + learningRate *currentSigma* currentInput;
            }
        }
        nnTest.backPropagation(mockOutputs, mockOutputsHidden);
        double[][] updatedInputToHiddenWeights = nnTest.getInputToHiddenWeights();
        double[][] updatedHiddenToOutWeights = nnTest.getHiddenToOutputWeights();

        for (int k =0; k<5; k++){
            double diff1 = Math.abs(expectedHiddenToOutWeights[k][0]-updatedHiddenToOutWeights[k][0]);
            if (diff1>0){
                Assert.fail();
            }
        }
        for(int l=0; l < expectedInputToHiddenUpdate.length ; l++){
            for(int l1 = 0; l1<expectedInputToHiddenUpdate[l].length; l1++){
                double diff2 = Math.abs(expectedInputToHiddenUpdate[l][l1]-updatedInputToHiddenWeights[l][l1]);
                if (diff2>0){
                    Assert.fail();
                }
            }
        }
    }

    @Test
    public void testBackPropagationUpdateInToHiddenForBipolar() {
        NeuralNet nnTest = new NeuralNet(bipolarInput, bipolarExpectedOutput, learningRate, momentum, noOfHiddenNeurons, false);
        nnTest.setInputToHiddenWeights(inToHidWeights2);
        nnTest.setHiddenToOutputWeights(hidToOutWeights2);

        double[] mockOutputsHidden = {1, 0.3, 0.4, 0.5, 0.6};
        double[] mockOutputs = {0.5};

        double f_prime_y = (1-Math.pow(mockOutputs[0],2)) / 2.0;
        double delta_y = f_prime_y*(bipolarExpectedOutput[0][0]-mockOutputs[0]);

        double[] f_prime_hidden = new double[5];
        double[] delta_hidden = new double[5];
        double[][] expectedHiddenToOutWeights = new double[5][1];

        //calculate expected weights from hidden to output, after first level of backpropagation
        for (int i=0; i< 5; i++){
            double currentW = hidToOutWeights2[i][0];
            expectedHiddenToOutWeights[i][0] = currentW + learningRate * delta_y * mockOutputsHidden[i];
        }

        //calculate expected delta, a.k.a error signal for each hidden neurons
        for (int i=0; i< 5; i++){
            f_prime_hidden[i] = (1 - Math.pow(mockOutputsHidden[i],2)) / 2.0;
            double currentW = expectedHiddenToOutWeights[i][0];
            delta_hidden[i] = currentW * delta_y * f_prime_hidden[i];
        }
        System.out.println(Arrays.toString(delta_hidden));

        //calculate the expected weight from input to hidden neurons, after backpropagation.
        double[][] expectedInputToHiddenUpdate = new double[3][4];
        for (int i1 = 0; i1 < 3; i1++) {
            for (int j = 0; j < 4; j++) {
                double currentW = inToHidWeights2[i1][j];
                double currentInput = bipolarInput[0][i1];
                double currentSigma = delta_hidden[j+1];
                expectedInputToHiddenUpdate[i1][j] = currentW + learningRate * currentSigma * currentInput;
            }
        }

        //actual implementation
        nnTest.backPropagation(mockOutputs, mockOutputsHidden);
        double[][] updatedInputToHiddenWeights = nnTest.getInputToHiddenWeights();
        double[][] updatedHiddenToOutWeights = nnTest.getHiddenToOutputWeights();

//        System.out.println("expectedInputToHiddenUpdate:");
//        for (double[] each:expectedInputToHiddenUpdate) {
//            System.out.println(Arrays.toString(each));
//        }
//        System.out.println("expectedHiddenToOutWeights:");
//        for (double[] each:expectedHiddenToOutWeights) {
//            System.out.println(Arrays.toString(each));
//        }
//        System.out.println("updatedInputToHiddenWeights");
//        for (double[] each:updatedInputToHiddenWeights) {
//            System.out.println(Arrays.toString(each));
//        }
//        System.out.println("updatedHiddenToOutWeights");
//        for (double[] each:updatedHiddenToOutWeights) {
//            System.out.println(Arrays.toString(each));
//        }

        for (int k =0; k<5; k++){
            double diff1 = Math.abs(expectedHiddenToOutWeights[k][0]-updatedHiddenToOutWeights[k][0]);
            if (diff1>0){
                Assert.fail();
            }
        }

        for(int l=0; l < expectedInputToHiddenUpdate.length ; l++){
            for(int l1 = 0; l1<expectedInputToHiddenUpdate[l].length; l1++){
                double diff2 = Math.abs(expectedInputToHiddenUpdate[l][l1]-updatedInputToHiddenWeights[l][l1]);
                if (diff2>0){
                    Assert.fail();
                }
            }
        }
    }
}
