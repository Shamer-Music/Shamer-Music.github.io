const myImage = document.querySelector("img");

myImage.addEventListener("click", () => { 
    const mySrc = myImage.getAttribute("src");
    if(mySrc === "images/Sham1.jpg") {
        myImage.setAttribute("src", "images/whats-up-there.jpg");
    } else {
        myImage.setAttribute("src", "images/Sham1.jpg");
    }
});

let myButton = document.querySelector("button");
let myHeading = document.querySelector("h1");

function setUserName() {
    const myName = prompt("Please enter your name.");
    if (!myName) {
        setUserName();
    } else {
    localStorage.setItem("name", myName);
    myHeading.textContent = `Shamer is cool, ${myName}`;
    }
}

if (!localStorage.getItem("name")) {
    setUserName();
} else {
    const storedName = localStorage.getItem("name");
    myHeading.textContent = `Shamer is cool, ${storedName}`;
}

myButton.addEventListener("click", () => {
    setUserName();
});
