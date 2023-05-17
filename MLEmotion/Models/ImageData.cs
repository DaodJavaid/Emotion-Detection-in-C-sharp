using Microsoft.ML.Data;
using MLEmotion.Models;

namespace MLEmotion.Models
{
    public class ModelInput
    {
        [ColumnName("Label"), LoadColumn(0)]
        public string Label { get; set; }

        [ColumnName("ImagePath"), LoadColumn(1)]
        public string ImagePath { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
