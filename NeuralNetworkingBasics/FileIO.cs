using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using VB = Microsoft.VisualBasic;

namespace NeuralNetworkingBasics
{
    class FileIO
    {
        static List<int> outputFields;
        private static string[][] ReadFile(string fileName)
        {
            outputFields = new List<int>();
            List<string[]> outputStream = new List<string[]>();
            try
            {
                using (VB.FileIO.TextFieldParser parser = new VB.FileIO.TextFieldParser(@fileName))
                {
                    parser.Delimiters = new string[] { "," };
                    string[] template = parser.ReadFields(); //dump template text

                    //mark output fields
                    for (int i = 0; i < template.Length; i++)
                    {
                        if(template[i].ToLower().Contains("output"))
                        {
                            outputFields.Add(i);
                        }
                    }
                    //store rest of the data in output stream
                    while (!parser.EndOfData)
                    {
                        outputStream.Add(parser.ReadFields());
                    }
                }
            }
            catch (IOException e)
            {
                Console.WriteLine("The file could not be read");
            }
            return outputStream.ToArray();
        }
        private static bool WriteFile(string fileName, string[] fileContents)
        {
            try
            {
                using (StreamWriter sw = new StreamWriter(fileName))
                {
                    int lastLine = fileContents.Length;
                    for (int line = 0; line < lastLine; line++)
                    {
                        sw.WriteLine(fileContents[line]);
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The file could not be written");
                return false;
            }
            //only if exception is not thrown
            return true;
        }
        private static double[] BuildDataSet(IEnumerable<string> inputSet)
        {
            List<double> buffer = new List<double>();

            foreach(string line in inputSet)
            {
                double d;
                double.TryParse(line, out d);
                buffer.Add(d);
            }

            return buffer.ToArray();
        }

        public static FileData UnwrapCSVFile(string fileName)
        {
            string[][] fileData = ReadFile(fileName);


            List<double[]> inputs = new List<double[]>();
            List<double[]> outputs = new List<double[]>();
            List<double[]> inputSet = new List<double[]>();

            for(int line = 0; line < fileData.Length; line++)
            {
                if(fileData[line].Length > outputFields[0])
                {
                    //this is an input-output learning pair
                    List<string> buffer_inputs = new List<string>();
                    List<string> buffer_outputs = new List<string>();

                    for( int field = 0; field < fileData[line].Length; field++)
                    {
                        if(outputFields.Contains(field))
                        {
                            buffer_outputs.Add(fileData[line][field]);
                        } else
                        {
                            buffer_inputs.Add(fileData[line][field]);
                        }
                    }

                    double[] buffer2_inputs = BuildDataSet(buffer_inputs);
                    double[] buffer2_outputs = BuildDataSet(buffer_outputs);

                    inputs.Add(buffer2_inputs);
                    outputs.Add(buffer2_outputs);

                } else
                {
                    //then it's just an input set
                    inputSet.Add(BuildDataSet(fileData[line]));
                }

            }

            IOBatch i = new IOBatch();
            i.InputList = inputs;
            i.OutputList = outputs;

            FileData returnData = new FileData();
            returnData.LearningData = i;
            returnData.InputSet = inputSet;

            return returnData;
        }
    }
    struct FileData
    {
        public IOBatch LearningData;
        public List<double[]> InputSet;
    }
}
