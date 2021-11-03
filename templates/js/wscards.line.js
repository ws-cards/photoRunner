	window.onload=function(){
		setTimeout(function(){
			window.scrollTo(0, 1);
		}, 100);		
			  setFun();  
	}

			var requestURLCardPrice = 'https://storage.googleapis.com/ws-cards.cloud/json/cardData.json';
			var requestMappingURL = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/cardDisplayMapping.json'
			var requestURLCardPricebyPreCode = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/cardData/';
			var requestURLCardTitle = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/cardTitle.json';
			var standardWURL = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/cardStandard_W.json';
			var standardSURL = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/cardStandard_S.json';
			var requestStandardW = new XMLHttpRequest();
			var requestStandardS = new XMLHttpRequest();			
			var requestPrice = new XMLHttpRequest();	
			var requestTitle = new XMLHttpRequest();	
			var requestMapping = new XMLHttpRequest();
			var mappingRep;
			  requestMapping.open('GET',requestMappingURL);
			  requestMapping.responseType = 'json';
			  requestMapping.send();	
			  requestMapping.onload = function() {console.log("debug:5");
			  mappingRep = requestMapping.response;
			  
			  }			  
			
			function setFun(){
			  //select 設定
			  var selectStandard = document.getElementById("cardStandard");
			  selectStandard.length = 1;
			  selectStandard.options[0].selected = true;	
			  
			  var selectPrice = document.getElementById("cardNumber"); 
			  selectPrice.length = 1;
			  selectPrice.options[0].selected = true;	
			  selectPrice.style.visibility = 'hidden';

			  var selectTitle = document.getElementById("cardTitle"); 
			  selectTitle.length = 1;
			  selectTitle.options[0].selected = true;	
			  
			  //request 設定
			  requestStandardW.open('GET', standardWURL);
			  requestStandardW.responseType = 'json';
			  requestStandardW.send();				  
			  requestStandardS.open('GET', standardSURL);
			  requestStandardS.responseType = 'json';
			  requestStandardS.send();	
			  
			  requestStandardW.onload = function(){
					var optgroupW = document.getElementById("Weiss");
			  		var cardsW = requestStandardW.response;
					for(var key in cardsW){	 
						var option = document.createElement("option");
						option.setAttribute("value",cardsW[key]);
						option.appendChild(document.createTextNode(key)); 
						optgroupW.appendChild(option);				
					}	
			  }
			  requestStandardS.onload = function(){
					var optgroupS = document.getElementById("Schwarz");				  
			  		var cardsS = requestStandardS.response;		
					for(var key in cardsS){	 
						var option = document.createElement("option");
						option.setAttribute("value",cardsS[key]);
						option.appendChild(document.createTextNode(key)); 
						optgroupS.appendChild(option);				
					}					
			  }
			  
			  requestPrice.open('GET', requestURLCardPrice);
			  requestPrice.responseType = 'json';
			  requestPrice.send();	
			
			  requestTitle.open('GET', requestURLCardTitle);
			  requestTitle.responseType = 'json';
			  requestTitle.send();
			
			  requestTitle.onload = function(){
				var cardsTitle = requestTitle.response;
				for(var key in cardsTitle){	 
					var option = document.createElement("option");
					option.setAttribute("value",key);
					option.appendChild(document.createTextNode(cardsTitle[key])); 
					selectTitle.appendChild(option);				
				}
			
			  }
			  
			  requestPrice.onload = function(){
				  var cards = requestPrice.response;
				  getCardData(cards,'BD/W54-070SSP','BD/W54-070SSP');			  
			  }
				var timer = setInterval(function(){
					if (document.getElementById('cardImg').complete){
					clearInterval(timer);
					console.log(document.getElementById('cardImg').complete)
					document.getElementById('overlay-1').style.display='none';	
					}
				}, 10);	
			}
			
			function changeStandard(){
			  var cardStandard=document.getElementById('cardStandard').value;
			  var cardStandardEle=document.getElementById('cardStandard');
			  var selectTitle = document.getElementById("cardTitle"); 
			  while (selectTitle.firstChild) {
				selectTitle.removeChild(selectTitle.firstChild);
			  }			  
			  
    		  requestTitle.open('GET', requestURLCardTitle);
			  requestTitle.responseType = 'json';
			  requestTitle.send();					
			  requestTitle.onload = function(){
				var cardsTitle = requestTitle.response;
				var cardStandardArray = cardStandard.split(",");
				for(var key in cardsTitle){	 

					var keyStr=key.substr(0,key.indexOf('/'));//2~3
					var keyStrLength=keyStr.length;

					var filtered = cardStandardArray.filter(function(value) {
						  return value === keyStr;
					});						
				    if(filtered==0){
						//double check
						continue;
					}

					var option = document.createElement("option");
					option.setAttribute("value",key);
					option.appendChild(document.createTextNode(cardsTitle[key])); 
					selectTitle.appendChild(option);				
				}
			  }		
				changeStandardAfterChangeNumber();			  
			}
						
			function removeTitle(){			
					document.getElementById('notuse').style.display='none';
			}				
			
			function changeStandardAfterChangeNumber(){
			  var selectPrice = document.getElementById("cardNumber"); 
			  selectPrice.style.visibility = 'visible';		
			  while (selectPrice.firstChild) {
				selectPrice.removeChild(selectPrice.firstChild);
			  }				
			  var cardTitle = document.getElementById('cardTitle').value;
			  var selectTitle = document.getElementById("cardTitle"); 			  
			  var option = document.createElement("option"); 
				  option.setAttribute("value",0);
				  option.appendChild(document.createTextNode("--選擇主題--")); 				  
				  selectTitle.appendChild(option);	
				  selectTitle.insertBefore(option,selectTitle.childNodes[0]);			 
			}
			
			function changeTitle(){	
			console.log("debug:1");
			  sortOption();console.log("debug:2");
			  //select 設定
			  var selectPrice = document.getElementById("cardNumber"); 
			  selectPrice.style.visibility = 'visible';
			  
			  selectPrice.length = 1;
			  selectPrice.options[0].selected = true;	
			  while (selectPrice.firstChild) {
				selectPrice.removeChild(selectPrice.firstChild);
			  }					  
			  
			  var cardTitle = document.getElementById('cardTitle').value;
			  			  
			  var cardTilteReplaceSpare = cardTitle.replace('/','_');
			  console.log(cardTitle+'->'+cardTilteReplaceSpare);
			  requestPrice.open('GET', requestURLCardPricebyPreCode + cardTilteReplaceSpare +'.json');
			  requestPrice.responseType = 'json';
			  requestPrice.send();				  
			  console.log("debug:3");
					  
			  console.log("debug:4");
				  requestPrice.onload = function() {

					var cards = requestPrice.response;
					  for(var key in cards){console.log("debug:6");
							if(key.indexOf('/')<0&&key.indexOf('S')==0){
										
												var option = document.createElement("option"); 
												option.setAttribute("value",key);
												option.appendChild(document.createTextNode(mappingRep[key])); 							
												selectPrice.appendChild(option);	
										console.log("debug:7");					
								}else{
									var option = document.createElement("option"); 
									option.setAttribute("value",key);
									option.appendChild(document.createTextNode(key)); 							
									selectPrice.appendChild(option);
								}					
						  }			
						 //重新排列option

						 sortOption();
						 selectPrice.options[0].selected=true;
						 changeNumber();
				}
			}
					
			
			function changeNumber(){	
				 var cardTitle = document.getElementById('cardTitle').value;
				 var cardTilteReplaceSpare = cardTitle.replace('/','_');
				console.log(cardTitle+'->'+cardTilteReplaceSpare);			 
			  document.getElementById('overlay-1').style.display='block';					
			  document.getElementById('overlay-2').style.display='block';				
				requestPrice.open('GET', requestURLCardPricebyPreCode + cardTilteReplaceSpare +'.json');
				requestPrice.responseType = 'json';
				requestPrice.send();
				requestPrice.onload = function() {
				  var cards = requestPrice.response;
				  var cardNumberSelect=document.getElementById('cardNumber');
				  var selectedIndex=cardNumberSelect.selectedIndex;
				  var cardNumberDisplay=cardNumberSelect.options[selectedIndex].text;				  
				  var internalCardNumber=cardNumberSelect.options[selectedIndex].value;		
				  getCardData(cards,internalCardNumber,cardNumberDisplay);
				}
				var timer = setInterval(function(){
					if (document.getElementById('cardImg').complete){
					clearInterval(timer);
					console.log(document.getElementById('cardImg').complete)
					document.getElementById('overlay-1').style.display='none';	
					}
				}, 10);			
			}
			

			
			
			/*繪圖區*/
			function getCardData(jsonObj,internalCardNumber,cardNum) {
				console.log("進入繪圖區:"+cardNum);
			  addPhoto(cardNum);
			  var cardInfo = jsonObj[internalCardNumber];
				/*		
				var cardNumber;
				if(internalCardNumber.indexOf(' ')>=0){
					cardNumber=internalCardNumber.substr(0,card_Num.indexOf(' '));
				}else{
					cardNumber=internalCardNumber;
				}*/
				var cardPriceUpDate=cardInfo['upddate'];
				var cardData=cardInfo['cardPrice'];
				 console.log("cardData:"+cardData); 
				const canvas = document.getElementById('myChart');
				const ctx = canvas.getContext('2d');
				const chart = new Chart(ctx, {
					responsive: true,
					// The type of chart we want to create
					type: 'line',

					// The data for our dataset
					data: {
						labels: cardPriceUpDate,
						datasets: [{
							label: cardNum,
							//fill:false,
							borderColor: 'rgb(255, 99, 132)',
							data: cardData
						}],
					},
					// Configuration options go here
					options: {
					tooltips: {
						mode: 'index',
						intersect: false,
					},
					hover: {
						mode: 'nearest',
						intersect: true
					},						
						scales:{
							xAxes: [{
								display: true,
								scaleLabel: {
									display: true,
									labelString: '日期'
								}
							}],
							yAxes: [{
								display: true,
								scaleLabel: {
									display: true,
									labelString: '價格(日幣)'
								}
							}]
						}
					
					}
				});		

				/*listener*/				
				var cardNumberListener = document.getElementById("cardNumber");
				cardNumberListener.addEventListener("change", function(){
					chart.destroy();
				});		
				var cardTitleListener = document.getElementById("cardTitle");	
				cardTitleListener.addEventListener("change", function(){
					chart.destroy();
				});		
				document.getElementById('overlay-2').style.display='none';					
			}

			
			/*加上圖片*/
			function addPhoto(cardNumberDisplay){
				var card_Num;
				if(cardNumberDisplay.indexOf(' ')>=0){
					card_Num=cardNumberDisplay.substr(0,cardNumberDisplay.indexOf(' '));
				}else{
					card_Num=cardNumberDisplay;				
				}

				var card_first=card_Num.substr(0,1);
				var card_second=card_Num.substr(0,card_Num.indexOf('-'));
					card_second=card_second.replace('/','_')
				var card_third=card_Num.replace('/','_');
					card_third=card_third.replace('-','_');	
				const cardImg = document.getElementById('cardImg');
				var urlCard="https://ws-tcg.com/wordpress/wp-content/cardimages/"+card_first.toLowerCase()+"/"+card_second.toLowerCase()+"/"+card_third.toLowerCase()+".png";
				//var urlCard="https://i.imgur.com/DKvx5hw.png";
				console.log(urlCard);
				cardImg.setAttribute("src",urlCard);
			}
			
		/*	
		Sort Option			
		*/
        function addOption(object, object2) { 
            each(object2, function(o, index) { 
                object.options[index] = o; 
            }) 
        } 
        function sortlist(sortName,isDesc) { 
            var what = document.getElementById(sortName); 
            this._options = map(what.options, function(o) { 
                return o; 
            }); 
            this._options.sort( function(a, b) { 
                if (a.text > b.text) { 
                    return isDesc == true ? 1 : -1; 
                } else { 
                    return isDesc == true ? -1 : 1; 
                } 
            }); 
            what.options.length = 0;// clear current options 
            addOption(what, this._options); 
        } 
        function map(object, callback, thisp) { 
            var ret = []; 
            each.call(thisp, object, function() { 
                ret.push(callback.apply(thisp, arguments)); 
            }); 
            return ret; 
        } 
        function each(object, callback) { 
            if (undefined === object.length) { 
                for ( var name in object) { 
                    if (false === callback(object[name], name, object)) 
                        break; 
                } 
            } else { 
                for ( var i = 0, len = object.length; i < len; i++) { 
                    if (i in object) { 
                        if (false === callback(object[i], i, object)) 
                            break; 
                    } 
                } 
            } 
        } 
        var sOrder = true; 
        function sortOption(){         
            if(sOrder){ 
                sOrder    = false; 
            }else{ 
                sOrder    = true; 
            } 
            sortlist("cardNumber",sOrder); 
        } 	
