using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkingBasics
{
    class CrossEntropy
    {
        static double Fn(double calculatedOutput, double realOutput)
        {
            double a = calculatedOutput;
            double y = realOutput;

            return -1 * y * Math.Log(a) - (1 - y) * Math.Log(1 - a);
        }

        static double Delta(double a, double y)
        {
            return a - y;
        }
    }
}
