
import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.util.simple.TrainingSetUtil;

/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming. This example shows how to construct an Encog neural network to
 * predict the output from the XOR operator. This example uses resilient
 * propagation (RPROP) to train the neural network. RPROP is the best general
 * purpose supervised training method provided by Encog.
 *
 * For the XOR example with RPROP I use 4 hidden neurons. XOR can get by on just
 * 2, but often the random numbers generated for the weights are not enough for
 * RPROP to actually find a solution. RPROP can have issues on really small
 * neural networks, but 4 neurons seems to work just fine.
 *
 * This example reads the XOR data from a CSV file. This file should be
 * something like:
 *
 * 0,0,0
 * 1,0,1
 * 0,1,1
 * 1,1,0
 */

public class XorHelloWorld
{

        public static void main(final String args[]) {

            if (args.length == 0) {
                System.out.println("Usage:\n\nXORCSV [xor.csv]");
            } else {
                final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(
                        CSVFormat.ENGLISH, args[0], false, 2, 1);
                final BasicNetwork network = EncogUtility.simpleFeedForward(2, 4,
                        0, 1, true);

                System.out.println();
                System.out.println("Training Network");
                EncogUtility.trainToError(network, trainingSet, 0.01);

                System.out.println();
                System.out.println("Evaluating Network");
                EncogUtility.evaluate(network, trainingSet);
            }
            Encog.getInstance().shutdown();
        }
}

