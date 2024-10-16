import { useState } from "react";

import "./App.css";
import Navbar from "./Components/Navbar/Navbar";
import Footer from "./Components/Footer/Footer";
import Search from "./Components/Search/Search";

function App() {
  return (
    <div>
      <Navbar />
       {/* <Info /> 
       <WhyUs />  
       <Services /> */}
      <Search />

      <Footer />
    </div>
  );
}

export default App;
