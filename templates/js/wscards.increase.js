			var requestURLCardPrice = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/report/yuyuPriceIncreaseReport.json';		
			var requestPrice = new XMLHttpRequest();	
			var requestMappingURL = 'https://storage.googleapis.com/divine-vehicle-292507.appspot.com/json/cardDisplayMapping.json';
			var requestMapping = new XMLHttpRequest();			
			var mappingRep;
			requestMapping.open('GET',requestMappingURL);
			requestMapping.responseType = 'json';
			requestMapping.send();	
			requestMapping.onload = function() {
			   console.log("debug:5");
			   mappingRep = requestMapping.response;
			}				
				
			function setFun(times){
			  //select 設定
			  var upTable = document.getElementById("upTable"); 
			  var downTable = document.getElementById("downTable"); 
			  //request 設定
			  requestPrice.open('GET', requestURLCardPrice);
			  requestPrice.responseType = 'json';
			  requestPrice.send();	
		
			
			  requestPrice.onload = function(){
				var cardsPrice = requestPrice.response;
				var upArray=new Array();
				var downArray=new Array();
				var arraySetLength=times					
				for(var key in cardsPrice){	 
					var cardPriceInfo = cardsPrice[key];
					var cardPriceMax = cardPriceInfo ['today'];
					var cardPriceMin = cardPriceInfo ['prePrice'];
					var statute = cardPriceInfo ['statute'];
					
					var firstDatePrice;
					var lastDatePrice;
					var spread;
                                        //沒有特別意義的交換，但是後面的變數要改
					if("upper" === statute){
						firstDatePrice =cardPriceMin;
						lastDatePrice = cardPriceMax;					
					}else{
						firstDatePrice = cardPriceMax;
						lastDatePrice = cardPriceMin ;	
					}
					spread=Math.round((((lastDatePrice-firstDatePrice)/firstDatePrice)*100)*100)/100;		
					
					if(upArray.length<arraySetLength||downArray.length<arraySetLength){
						console.log("漲價牌組:"+upArray.length);
						console.log("跌價牌組:"+downArray.length);
						console.log("限制長度:"+arraySetLength);
						if("upper" === statute && upArray.length<arraySetLength){
							upArray.push({cardSpread:spread,cardNo:key,price:lastDatePrice});
						}else if("downer" === statute && downArray.length<arraySetLength){
							downArray.push({cardSpread:spread,cardNo:key,price:firstDatePrice});
							//console.log("第一次:"+key+":"+firstDatePrice+":"+spread);
						}	
					}else{
						if("upper" === statute){
							upArray.push({cardSpread:spread,cardNo:key,price:lastDatePrice});
							upArray.sort(function(a, b) {
								return a.cardSpread < b.cardSpread ? 1: -1;
							});
							upArray.pop();					
						}else if("downer" === statute){//console.log("連續進入:"+key+":"+firstDatePrice+":"+spread);
							downArray.push({cardSpread:spread,cardNo:key,price:firstDatePrice});
							downArray.sort(function(a, b) {
								return a.cardSpread < b.cardSpread ? 1: -1;
							});
							downArray.pop();						
						}
					}
				}
				settingUpTable(upArray);
				settingDownTable(downArray);
				//取消loading
				document.getElementById('overlay1').style.display='none';	
				document.getElementById('overlay2').style.display='none';					
			  }

			}
			function settingUpTable(upArray){
				for(var key in upArray ){
					var cardNo=upArray[key].cardNo;
					var spread=upArray[key].cardSpread;
					var price=upArray[key].price;
						var tr = document.createElement("tr");
						//卡號稀有度價錢幅度
						var tdCardNo = document.createElement("td");
						var tdPrice = document.createElement("td");
						var tdRange = document.createElement("td");
						if(cardNo.indexOf('/')<0&&cardNo.indexOf('S')==0){
							cardNo = mappingRep[cardNo];
						}
						tdCardNo.innerHTML = cardNo;
						tdPrice.innerHTML = price;
						tdRange.innerHTML = "<small class='text-success mr-1' style='font-family: 'Noto Sans TC', sans-serif;font-size:10px;'><i class='fas fa-arrow-up' ></i>"+spread+"%</small>";

						tr.appendChild(tdCardNo);  
						tr.appendChild(tdPrice); 
						tr.appendChild(tdRange); 
						upTable.appendChild(tr);	
				}					
			}
			function settingDownTable(downArray){
				for(var key in downArray ){
					var cardNo=downArray[key].cardNo;
					var spread=downArray[key].cardSpread;
					var price=downArray[key].price;					
						var tr = document.createElement("tr");
						//卡號稀有度價錢幅度
						var tdCardNo = document.createElement("td");
						var tdPrice = document.createElement("td");
						var tdRange = document.createElement("td");
						if(cardNo.indexOf('/')<0&&cardNo.indexOf('S')==0){
							cardNo = mappingRep[cardNo];
						}
						tdCardNo.innerHTML = cardNo;
						tdPrice.innerHTML = price;
						tdRange.innerHTML = "<small class='text-warning mr-1' style='font-family: 'Noto Sans TC', sans-serif;font-size:10px;'><i class='fas fa-arrow-down' ></i>"+spread+"%</small>";
						tr.appendChild(tdCardNo); 
						tr.appendChild(tdPrice); 
						tr.appendChild(tdRange); 
						downTable.appendChild(tr);	
				}					
			}
