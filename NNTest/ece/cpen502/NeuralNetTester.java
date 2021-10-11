package ece.cpen502;

import org.junit.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Assert;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NeuralNetTester {
    NeuralNet testBinaryNN_complex;
//    NeuralNet testBinaryNN_easy;
    NeuralNet testBipolarNN;
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
    public void testSigmoid() {
        // TODO
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        double x = 0.1;
        double expecteRes = 1/(1+Math.exp(-x));
        assertEquals(expecteRes, nnTest.sigmoid(x), "sigmoid calculation should be correct");
    }

    @Test
    public void testForwardToHidden() {
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        nnTest.setInputToHiddenWeights(inToHidWeights2);
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
    public void testForwardToOutput() {
        NeuralNet nnTest = new NeuralNet(binaryInput2, binaryExpectedOutput2, learningRate, momentum, noOfHiddenNeurons, true);
        nnTest.setHiddenToOutputWeights(hidToOutWeights2);
        double[] mockOutputsHidden = {1, 0.3, 0.4, 0.5, 0.6};
        double s = 1*0.1+ 0.3*0.2 + 0.4*(-0.3) + 0.5*(-0.4)+0.6*0.5;
        double y = nnTest.sigmoid(s);
        double[] expectedOutputs = {y};
        Assert.assertArrayEquals(expectedOutputs, nnTest.forwardToOutput(mockOutputsHidden), 0);
    }
//
    @Test
    public void testBackPropagationUpdateInToHidden() {
        // TODO
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
        int a = 2;

        for (int k =0; k<5; k++){
            double diff1 = Math.abs(expectedHiddenToOutWeights[k][0]-updatedHiddenToOutWeights[k][0]);
            if (diff1>0.01){
                Assert.fail();
            }
        }
        for(int l=0; l < expectedInputToHiddenUpdate.length ; l++){
            for(int l1 = 0; l1<expectedInputToHiddenUpdate[l].length; l1++){
                double diff2 = Math.abs(expectedInputToHiddenUpdate[l][l1]-updatedInputToHiddenWeights[l][l1]);
                if (diff2>0.01){
                    Assert.fail();
                }
            }
        }
    }
}
