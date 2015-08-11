using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkingBasics
{
    class Program
    {
        static void Main(string[] args)
        {
            //create and design network
            Network n = new Network(1, 3, 1);

            //organize input output pairs
            List<double[]> inputs;
            List<double[]> outputs;
            List<double[]> inputSet;

            FileData info = FileIO.UnwrapCSVFile("./data.csv");
            inputs = info.LearningData.InputList;
            outputs = info.LearningData.OutputList;
            inputSet = info.InputSet;

            int batchSize = 64;

            //tell network to learn from inputs
            n.GradientDescent(inputs, outputs, inputs.Count/batchSize, batchSize, 1.7, 100);

            //run the following inputs and output the results
            foreach (double[] i in inputSet)
                Print(n.Run(i));


            Console.Read();
        }

        static void Print(double[] vector)
        {
            Console.Write("<");
            foreach (double v in vector)
                Console.Write("{0}, ", v);
            Console.Write(">\n");
        }
        static void Print(IEnumerable<double[]> vectors)
        {
            foreach (double[] v in vectors)
                Print(v);
        }
    }
}
