/**
 * TAI Score Display Component
 * Shows the comprehensive TAI scoring breakdown with visual indicators
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Trophy, 
  Shield, 
  Shuffle, 
  Brain, 
  Droplets, 
  DollarSign,
  Star 
} from 'lucide-react';

import { TAIScore } from '../../types/enhanced-analysis';

interface TAIScoreDisplayProps {
  scores: TAIScore;
}

interface ScoreItemProps {
  label: string;
  value: number;
  icon: React.ReactNode;
  description?: string;
}

const ScoreItem: React.FC<ScoreItemProps> = ({ label, value, icon, description }) => {
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 70) return 'bg-blue-500';
    if (score >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getTextColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 70) return 'text-blue-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium">
          {icon}
          {label}
        </div>
        <span className={`text-sm font-bold ${getTextColor(value)}`}>
          {value.toFixed(1)}
        </span>
      </div>
      <Progress 
        value={value} 
        className="h-2"
      />
      {description && (
        <p className="text-xs text-gray-500">{description}</p>
      )}
    </div>
  );
};

export const TAIScoreDisplay: React.FC<TAIScoreDisplayProps> = ({ scores }) => {
  const getOverallScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 70) return 'text-blue-600 bg-blue-50 border-blue-200';
    if (score >= 60) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getGradeBadgeColor = (grade: string) => {
    switch (grade) {
      case 'A': return 'bg-green-100 text-green-800';
      case 'B': return 'bg-blue-100 text-blue-800';
      case 'C': return 'bg-yellow-100 text-yellow-800';
      case 'D': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Star className="w-5 h-5 text-yellow-500" />
          TAI Portfolio Score
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Score */}
        <div className={`p-4 rounded-lg border-2 ${getOverallScoreColor(scores.overall_score)}`}>
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-3xl font-bold">
                {scores.overall_score.toFixed(1)}
              </div>
              <div className="text-sm opacity-75">
                Overall TAI Score
              </div>
            </div>
            <Badge className={getGradeBadgeColor(scores.grade)} variant="secondary">
              Grade {scores.grade}
            </Badge>
          </div>
          <p className="text-sm">{scores.description}</p>
        </div>

        {/* Component Scores */}
        <div className="space-y-4">
          <h4 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">
            Score Breakdown
          </h4>
          
          <div className="grid gap-4">
            <ScoreItem
              label="Performance"
              value={scores.performance_score}
              icon={<Trophy className="w-4 h-4 text-yellow-600" />}
              description="Returns, volatility, and risk-adjusted performance"
            />
            
            <ScoreItem
              label="Risk Management"
              value={scores.risk_management_score}
              icon={<Shield className="w-4 h-4 text-blue-600" />}
              description="Drawdown control, beta, and risk metrics"
            />
            
            <ScoreItem
              label="Diversification"
              value={scores.diversification_score}
              icon={<Shuffle className="w-4 h-4 text-purple-600" />}
              description="Sector spread, correlation, and concentration risk"
            />
            
            <ScoreItem
              label="ML Confidence"
              value={scores.ml_confidence_score}
              icon={<Brain className="w-4 h-4 text-pink-600" />}
              description="Model prediction confidence and reliability"
            />
            
            <ScoreItem
              label="Liquidity"
              value={scores.liquidity_score}
              icon={<Droplets className="w-4 h-4 text-cyan-600" />}
              description="Market liquidity and ease of exit"
            />
            
            <ScoreItem
              label="Cost Efficiency"
              value={scores.cost_efficiency_score}
              icon={<DollarSign className="w-4 h-4 text-green-600" />}
              description="Transaction costs and expense ratios"
            />
          </div>
        </div>

        {/* Score Legend */}
        <div className="pt-4 border-t border-gray-200">
          <h5 className="text-xs font-semibold text-gray-600 mb-2">Score Range</h5>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span>80-100: Excellent</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <span>70-79: Good</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-yellow-500 rounded"></div>
              <span>60-69: Average</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>0-59: Needs Work</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}; 