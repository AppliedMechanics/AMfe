function loader() {
    setWelcomeScreen();        
}

function setWelcomeScreen(){
    $("#amfe-container").css("height", $(window).height()-$("#main-nav").height());
    $("#amfe-container").css("background-size","contain");
    $("#welcome-continue").css("top", $("#amfe-container").height() - 100);
}

function animateWelcomeContinue(){
        var scrollhoehe = $("#amfe-container").height() + $("#main-nav").height();
        $("html, body").animate({ scrollTop: scrollhoehe },500);
}