import React, { useRef, useEffect } from 'react';
import { Trash2, Zap } from 'lucide-react';
import './DrawingCanvas.scss';

const DrawingCanvas = ({ onPredict, isLoading, onClear }) => {
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const lastPosRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    initCanvas();
  }, []);

  const initCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';
  };

  const getMousePos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e) => {
    isDrawingRef.current = true;
    lastPosRef.current = getMousePos(e);
  };

  const draw = (e) => {
    if (!isDrawingRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const currentPos = getMousePos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
    ctx.lineTo(currentPos.x, currentPos.y);
    ctx.stroke();
    
    lastPosRef.current = currentPos;
  };

  const stopDrawing = () => {
    isDrawingRef.current = false;
  };

  const handleClear = () => {
    initCanvas();
    onClear();
  };

  const handlePredict = () => {
    const canvas = canvasRef.current;
    const imageData = canvas.toDataURL('image/png');
    onPredict(imageData);
  };

  return (
    <div className="drawing-canvas">
      <h2 className="drawing-canvas__title">✏️ Draw Here</h2>
      
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className="drawing-canvas__canvas"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />

      <div className="drawing-canvas__controls">
        <button
          onClick={handleClear}
          className="drawing-canvas__btn drawing-canvas__btn--secondary"
          disabled={isLoading}
        >
          <Trash2 size={20} />
          Clear
        </button>
        <button
          onClick={handlePredict}
          className="drawing-canvas__btn drawing-canvas__btn--primary"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <div className="spinner" />
              Analyzing...
            </>
          ) : (
            <>
              <Zap size={20} />
              Predict
            </>
          )}
        </button>
      </div>

      <div className="drawing-canvas__instructions">
        <p className="instructions__title">💡 Tips:</p>
        <ul className="instructions__list">
          <li>✓ Draw large and centered</li>
          <li>✓ Use your mouse or touch screen</li>
          <li>✓ Click "Predict" to see results</li>
        </ul>
      </div>
    </div>
  );
};

export default DrawingCanvas;