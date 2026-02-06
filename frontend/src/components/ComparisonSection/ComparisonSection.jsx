import React from 'react';
import './ComparisonSection.scss';

const ComparisonSection = ({ mlpPrediction, cnnPrediction }) => {
  if (!mlpPrediction || !cnnPrediction) return null;

  const winner = cnnPrediction.confidence > mlpPrediction.confidence ? 'CNN' : 'MLP';
  const agree = mlpPrediction.digit === cnnPrediction.digit;

  return (
    <div className="comparison">
      <h2 className="comparison__title">📊 Model Comparison</h2>
      <div className="comparison__grid">
        
        <div className="comparison__card comparison__card--mlp">
          <div className="card__label">MLP Prediction</div>
          <div className="card__digit">{mlpPrediction.digit}</div>
          <div className="card__confidence">
            {(mlpPrediction.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="comparison__card comparison__card--cnn">
          <div className="card__label">CNN Prediction</div>
          <div className="card__digit">{cnnPrediction.digit}</div>
          <div className="card__confidence">
            {(cnnPrediction.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="comparison__card comparison__card--winner">
          <div className="card__label">Winner</div>
          <div className="card__digit">{winner}</div>
          <div className="card__status">
            {agree ? '✓ Both Agree!' : '⚠️ Different Results'}
          </div>
        </div>

      </div>
    </div>
  );
};

export default ComparisonSection;