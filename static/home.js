
function hand(){
	var s=prompt("Enter the sentence");
	var x=0, n=s.length;
	for(let i=0; i<n; i++){
		var p=document.createElement("img");
		var t="static/";
		if(s[i]==" ") {
			t+="SPACE";
		}
		else {
			t+=s[i];
		}
		t+=".jpg";
		p.src=t;
		document.body.appendChild(p);
	}
}
