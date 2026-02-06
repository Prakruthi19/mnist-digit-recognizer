import React from 'react';
import './Header.scss';

const Header = () => {
  return (
    <header className="header">
      <h1 className="header__title">🎨 MNIST Digit Recognition</h1>
      <p className="header__subtitle">
        Draw a digit and watch AI recognize it in real-time!
      </p>
      <p className="header__description">
        Compare MLP vs CNN Performance
      </p>
    </header>
  );
};

export default Header;