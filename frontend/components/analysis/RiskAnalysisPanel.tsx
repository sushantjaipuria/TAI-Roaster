/**
 * Risk Analysis Panel Component
 * Displays comprehensive risk metrics and warnings
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield } from 'lucide-react';

import { PerformanceMetrics, RiskWarning, RiskLevel } from '../../types/enhanced-analysis';

interface RiskAnalysisPanelProps {
  performanceMetrics: PerformanceMetrics;
  riskWarnings: RiskWarning[];
  riskLevel: RiskLevel;
}

export const RiskAnalysisPanel: React.FC<RiskAnalysisPanelProps> = ({ 
  performanceMetrics, 
  riskWarnings, 
  riskLevel 
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-500" />
          Risk Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600">Risk Level</div>
              <div className="text-lg font-semibold">{riskLevel}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Volatility</div>
              <div className="text-lg font-semibold">{performanceMetrics.volatility.toFixed(2)}%</div>
            </div>
          </div>
          {riskWarnings.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Risk Warnings</h4>
              {riskWarnings.map((warning, index) => (
                <div key={index} className="text-sm text-red-600">
                  {warning.message}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}; 