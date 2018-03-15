
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
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

public class Weather
{

        public static void main(final String args[]) {

            if (args.length == 0) {
                System.out.println("Usage:\n\nWeather [weather.csv]");
            } else {
                BasicNetwork network = new BasicNetwork();
                network.addLayer(new BasicLayer(null,true,3));
                network.addLayer(new BasicLayer(new ActivationSigmoid(),true,5));
                network.addLayer(new BasicLayer(new ActivationSigmoid(),true,1));
                network.getStructure().finalizeStructure();
                network.reset();

                final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(
                        CSVFormat.ENGLISH, args[0], true, 3, 1);
                final Backpropagation train = new Backpropagation(network, trainingSet);

                //final BackPropagation train = new ResilientPropagation(network, trainingSet);

                int epoch = 1;

                do {
                    train.iteration();
                    System.out.println("Epoch #" + epoch + " Error:" + train.getError());
                    epoch++;
                } while(train.getError() > 0.01);
                train.finishTraining();
                BasicMLData data = new BasicMLData(3);
                data.setData(new double[]{0.3302469136,0.1125265393,0.5142857143});
                MLData d = network.compute(new BasicMLData(3));
                System.out.println(d);
            }

            Encog.getInstance().shutdown();
        }
}

