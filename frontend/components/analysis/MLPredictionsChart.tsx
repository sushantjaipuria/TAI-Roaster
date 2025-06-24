/**
 * ML Predictions Chart Component
 * Visualizes machine learning model predictions for individual stocks
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Brain, TrendingUp, TrendingDown } from 'lucide-react';

import { MLPrediction } from '../../types/enhanced-analysis';

interface MLPredictionsChartProps {
  predictions: MLPrediction[];
}

export const MLPredictionsChart: React.FC<MLPredictionsChartProps> = ({ predictions }) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getPredictionColor = (prediction: number) => {
    if (prediction > 0.1) return 'text-green-600';
    if (prediction < -0.1) return 'text-red-600';
    return 'text-gray-600';
  };

  const getPredictionIcon = (prediction: number) => {
    if (prediction > 0.05) return <TrendingUp className="w-4 h-4 text-green-600" />;
    if (prediction < -0.05) return <TrendingDown className="w-4 h-4 text-red-600" />;
    return <div className="w-4 h-4 bg-gray-400 rounded-full"></div>;
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-500" />
          ML Model Predictions
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {predictions.filter(p => p.ensemble_prediction > 0.05).length}
              </div>
              <div className="text-sm text-gray-600">Bullish Signals</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">
                {predictions.filter(p => Math.abs(p.ensemble_prediction) <= 0.05).length}
              </div>
              <div className="text-sm text-gray-600">Neutral</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {predictions.filter(p => p.ensemble_prediction < -0.05).length}
              </div>
              <div className="text-sm text-gray-600">Bearish Signals</div>
            </div>
          </div>

          {/* Individual Predictions */}
          <div className="space-y-3">
            <h4 className="font-semibold text-sm text-gray-700">Individual Stock Predictions</h4>
            
            {predictions.map((prediction, index) => (
              <div key={index} className="border rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getPredictionIcon(prediction.ensemble_prediction)}
                    <div>
                      <div className="font-semibold">{prediction.ticker}</div>
                      <div className="text-sm text-gray-600">Ensemble Prediction</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${getPredictionColor(prediction.ensemble_prediction)}`}>
                      {formatPercentage(prediction.ensemble_prediction)}
                    </div>
                    <Badge className={getConfidenceColor(prediction.ensemble_confidence)} variant="secondary">
                      {(prediction.ensemble_confidence * 100).toFixed(0)}% confidence
                    </Badge>
                  </div>
                </div>

                {/* Model Breakdown */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600">XGBoost</div>
                    <div className="font-semibold">{formatPercentage(prediction.xgboost_prediction)}</div>
                  </div>
                  {prediction.lightgbm_prediction && (
                    <div>
                      <div className="text-gray-600">LightGBM</div>
                      <div className="font-semibold">{formatPercentage(prediction.lightgbm_prediction)}</div>
                    </div>
                  )}
                  {prediction.catboost_prediction && (
                    <div>
                      <div className="text-gray-600">CatBoost</div>
                      <div className="font-semibold">{formatPercentage(prediction.catboost_prediction)}</div>
                    </div>
                  )}
                  <div>
                    <div className="text-gray-600">NGBoost (Mean ± Std)</div>
                    <div className="font-semibold">
                      {formatPercentage(prediction.ngboost_mean)} ± {formatPercentage(prediction.ngboost_std)}
                    </div>
                  </div>
                </div>

                {/* Prediction Interpretation */}
                <div className="pt-2 border-t border-gray-100">
                  <div className="text-xs text-gray-600">
                    <strong>Interpretation:</strong>{' '}
                    {prediction.ensemble_prediction > 0.1 ? (
                      "Strong positive momentum expected"
                    ) : prediction.ensemble_prediction > 0.05 ? (
                      "Moderate positive momentum expected"
                    ) : prediction.ensemble_prediction < -0.1 ? (
                      "Strong negative momentum expected"
                    ) : prediction.ensemble_prediction < -0.05 ? (
                      "Moderate negative momentum expected"
                    ) : (
                      "Sideways movement expected"
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Model Information */}
          <div className="pt-4 border-t border-gray-200">
            <h5 className="text-xs font-semibold text-gray-600 mb-2">About ML Predictions</h5>
            <div className="text-xs text-gray-500 space-y-1">
              <p>• Predictions represent expected short-term price movement (1-3 months)</p>
              <p>• Ensemble model combines multiple algorithms for better accuracy</p>
              <p>• Confidence scores indicate model certainty in predictions</p>
              <p>• Higher confidence predictions are generally more reliable</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}; 