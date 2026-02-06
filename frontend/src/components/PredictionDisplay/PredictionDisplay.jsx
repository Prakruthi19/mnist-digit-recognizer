import React from 'react';
import ConfidenceChart from '../ConfidenceChart/ConfidenceChart';
import './PredictionDisplay.scss';

const PredictionDisplay = ({ prediction, icon: Icon, title, color, modelType }) => {
  return (
    <div className={`prediction-display prediction-display--${modelType}`}>
      <div className="prediction-display__header">
        <Icon className="prediction-display__icon" style={{ color }} />
        <h2 className="prediction-display__title">{title}</h2>
      </div>
      
      <div className="prediction-display__result">
        <div className="result__digit" style={{ color }}>
          {prediction ? prediction.digit : '?'}
        </div>
        <div className="result__confidence">
          {prediction 
            ? `${(prediction.confidence * 100).toFixed(2)}% confident`
            : 'Draw a digit to start'}
        </div>
      </div>

      {prediction && (
        <ConfidenceChart predictions={prediction} color={color} />
      )}
    </div>
  );
};

export default PredictionDisplay;