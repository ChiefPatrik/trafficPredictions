import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="bg-gray-300 text-black p-4 shadow-md fixed w-full top-0 z-20 border border-gray-200 h-16">
      <div className="container mx-auto flex justify-between items-center">
        <nav className="flex justify-between w-full">
          {/* <div className='hover:bg-gray-500 w-1/2 h-1/2'> */}
            <Link to="/" className="hover:underline flex-1 text-center">
              PREDICTIONS
            </Link>
          {/* </div> */}

          <Link to="/admin" className="hover:underline flex-1 text-center">
            ADMIN PANEL
          </Link>
        </nav>
      </div>
    </header>
  //   <header className="relative bg-gray-300 text-black p-4 shadow-md fixed w-full top-0 z-20 border border-gray-200 h-16">
  //   <div className="container mx-auto flex justify-between items-center">
  //     <nav className="relative flex justify-between w-full">
  //       <Link to="/" className="hover:underline flex-1 text-center">
  //         PREDICTIONS
  //       </Link>
  //       <Link to="/admin" className="hover:underline flex-1 text-center">
  //         ADMIN PANEL
  //       </Link>
  //     </nav>
  //     <div className="absolute inset-0">
  //       <div className="absolute inset-0 bg-transparent hover:bg-gray-400 transition-all duration-300" />
  //       <div className="absolute inset-0 bg-transparent hover:bg-gray-400 transition-all duration-300" />
  //     </div>
  //   </div>
  // </header>
  );
};

export default Header;
