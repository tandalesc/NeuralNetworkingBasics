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
            Network n = new Network(5, 3, 1);

            //organize input output pairs
            List<double[]> inputs;
            List<double[]> outputs;
            List<double[]> inputSet;

            FileData info = FileIO.UnwrapCSVFile("data.csv");
            inputs = info.LearningData.InputList;
            outputs = info.LearningData.OutputList;
            inputSet = new List<double[]>();

            inputSet.Add(new double[]
                {1, 1, 1, 1, 1}
                );
            inputSet.Add(new double[]
                {0, 0, 0, 0, 1}
                );
            inputSet.Add(new double[]
                {0, 1, 1, 0.5, 0}
                );
            inputSet.Add(new double[]
                {0, 1, 1, 1, 1}
                );
            inputSet.Add(new double[]
                {1, 0, 0, 0, 1}
                );
            inputSet.Add(new double[]
                {1, 1, 1, 0.5, 0}
                );


            int batchSize = 16;

            //tell network to learn from inputs
            n.GradientDescent(inputs, outputs, 30, batchSize, 0.2, 0.295, 0.07);

            //run the following inputs and output the results
            foreach (double[] i in inputSet)
                Print(n.Run(i).Take(outputs[0].Length).ToArray());


            Console.Read();
        }

        static void Print(double[] vector)
        {
            Console.Write("<");
            foreach (double v in vector)
                Console.Write("{0}, ", Round(v));
            Console.Write(">\n");
        }
        static void Print(IEnumerable<double> vectors)
        {
            foreach (double v in vectors)
                Console.Write(v);
        }
        static int Round(double d)
        {
            return (d > 0.5) ? 1 : 0;
        }
    }
}
