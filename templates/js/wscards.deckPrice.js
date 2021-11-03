			var requestURLDeckLogDeck = 'https://divine-vehicle-292507.dt.r.appspot.com/api/deckLogInfo.jsp?deckcode=';	
			var requestDeckLog = new XMLHttpRequest();				
			
			window.onload=function(){
				setTimeout(function(){
					window.scrollTo(0, 1);
				}, 100);				
				document.getElementById('second-content').style.display='none';
				var today = new Date();
				var hours = today.getHours();
				var mins = today.getMinutes();
				if(hours === 10 && (mins < 59 && mins> 0)){
					Swal.fire({
					  icon: 'error',
					  title: '喔不...',
					  text: '現在為每日10:00-11:00為系統維護時間!',
					})					
					document.getElementById('deckCode').disabled = true;
					document.getElementById('buttonSearch').disabled = true;
					
					console.log("10:00-11:00為系統維護時間");
				}
			  //setFun();  
			}
			
			
			$('#exampleModalCenter').on('hidden.bs.modal', function (e) {
			 // $('#exampleModalCenter').modal('dispose');
				$('#exampleModalCenter').hide();
			})
			
			
			function deckCal(){
				document.getElementById('overlay-1').style='display';
				document.getElementById('overlay-2').style='display';					
				var oform = document.forms["calForm"];
				var deckCode = oform.elements["deckCode"].value.trim();
				var deckCodeLength=deckCode.length;
				if(deckCodeLength===4){
						Swal.fire({
						  title: '要查詢DeckLog的日文版還是英文版?',
						  icon: 'warning',
						  showDenyButton: true,
						  confirmButtonText: '日文',
						  denyButtonText: '英文',
						}).then((result) => {
						  if (result.isConfirmed) {
							//日文
							 console.log(deckCode); 
							removeTable();
							setDeckPrice(deckCode);		  
						  } else if (result.isDenied) {
							deckCode = "en_"+deckCode;
							 console.log(deckCode);
							removeTable();
							setDeckPrice(deckCode);								  
						  }
						})					
				}else{
							removeTable();
							setDeckPrice(deckCode);					
				}				

			}
			
			function setDeckPrice(deckCode){
			 try{
			  //request 設定
			  requestDeckLog.open('POST', requestURLDeckLogDeck+deckCode);
			  requestDeckLog.responseType = 'json';
			  requestDeckLog.send();
			
			  requestDeckLog.onload = function(){	
	
				var deckLockInfo = requestDeckLog.response;
				console.log(deckLockInfo);
				if('NODATA'.indexOf(deckLockInfo)>=0||deckLockInfo==null){
								Swal.fire({
								  icon: 'error',
								  title: '喔不...',
								  text: '找不到資料，有可能是輸入錯誤，可利用聯繫右下方Messenger回報!'
								})							
					document.getElementById('overlay-1').style.display='none';				
				}else if('Exception'.indexOf(deckLockInfo)>=0){
								Swal.fire({
								  icon: 'error',
								  title: '喔不...',
								  text: '網站發生不明錯誤，請利用右下角Messenger回報'
								})							
					document.getElementById('overlay-1').style.display='none';					
				}else{					
					var deckGameTitle = deckLockInfo['deckGameTitle'];
					var deckSource = deckLockInfo['deckSource'];
					var deckName = deckLockInfo['deckName'];
					var deckId = deckLockInfo['deckId'];
					var deckWSTitle = deckLockInfo['deckWSTitle'];
					var deckCardList = deckLockInfo['cardList'];
					var deckPriceDateList = deckCardList[0];
					var deckUpdatePriceDate = deckPriceDateList['uppdate'];
					console.log(deckUpdatePriceDate);
					if(!(deckGameTitle==='2'||deckGameTitle==='5'||deckGameTitle==='12'||deckGameTitle==='122')){
						switch(deckGameTitle){
							case '5':
								alert('你輸入的是Reバース的牌組!');
								document.getElementById('overlay-1').style.display='none';										
								break;
							case '1':
								Swal.fire({
								  icon: 'error',
								  title: '喔不...',
								  text: '現在還不支援VG的牌組喔!'
								})							
								document.getElementById('overlay-1').style.display='none';	
								break;
							case '9999':
								Swal.fire({
								  icon: 'error',
								  title: '喔不...',
								  text: '現在是每日 10:00-11:00 為 系統維護時間!'
								})									
								document.getElementById('overlay-1').style.display='none';	
								break;								
							default:
								Swal.fire({
								  icon: 'error',
								  title: '喔喔...',
								  text: '不是WS或是rebirth的牌組喔!'
								})		
								document.getElementById('overlay-1').style.display='none';	
						}	
					}else{
						document.getElementById('second-content').style='display';						
						document.getElementById('deckCodeDisplay').innerHTML='デッキコード: '+deckId;
						document.getElementById('deckTitleName').innerHTML='主題名稱:'+deckWSTitle;
						document.getElementById('deckName').innerHTML='牌組名稱:'+deckName;	
						document.getElementById('deckSource').innerHTML=deckSource;
						document.getElementById('deckSource').style.display='none';
						var today = new Date();
						var month = today.getMonth()+1;
						var date = today.getDate();
						var year = today.getFullYear();
						document.getElementById('searchDate').innerHTML="製表日期:"+year+"/"+month+"/"+date;
						document.getElementById('updatePriceDate').innerHTML="價格更新日期:"+deckUpdatePriceDate;
						calculateDeckPrice(deckCardList,deckGameTitle);
					}
				
				}
			  }
			 } 
			 catch(e){
				console.log("有誤");
				console.log(e);
			 }		 
			}
			
			function calculateDeckPrice(deckCardList,deckGameTitle){
			
				var deckTableHead=document.getElementById('deckTable');
				var thead = document.createElement("thead");
				var thead_tr = document.createElement("tr");
				var thead_tr_th_cardno = document.createElement("th");
				var thead_tr_th_cardrare = document.createElement("th");
				var thead_tr_th_cardprice = document.createElement("th");
				var thead_tr_th_cardnum = document.createElement("th");
				var thead_tr_th_cardtotal = document.createElement("th");
				var tbody = document.createElement("tbody");
				
				thead_tr_th_cardno.innerHTML="卡號";
				thead_tr_th_cardrare.innerHTML="稀有度";
				thead_tr_th_cardprice.innerHTML="單價";
				thead_tr_th_cardnum.innerHTML="數量";
				thead_tr_th_cardtotal.innerHTML="總和";
				thead_tr.appendChild(thead_tr_th_cardno);
				thead_tr.appendChild(thead_tr_th_cardrare);
				thead_tr.appendChild(thead_tr_th_cardprice);
				thead_tr.appendChild(thead_tr_th_cardnum);
				thead_tr.appendChild(thead_tr_th_cardtotal);
				thead.appendChild(thead_tr);	
				tbody.setAttribute("id","tableCardBody");
				
				deckTableHead.appendChild(thead);
				deckTableHead.appendChild(tbody);
				
				
				var getCardListFirstContent = deckCardList[0];
				var getFirstCardNumber = getCardListFirstContent ['cardNumber'];	
				var preCode=getFirstCardNumber.substr(0,getFirstCardNumber.indexOf('-'));
				var preCode_clean=preCode.replace('/','_')			
				var totalPrice=0;
				var totalNum=0;

					for(var cardkey in deckCardList){
						var cardList=deckCardList[cardkey];
						var cardNumber=cardList['cardNumber'];
						var cardTimes=cardList['cardTime'];
						var internalCardNumber=cardList['internalCardNumber'];
						var cardName=cardList['cardName'];
						var uppdate=cardList['uppdate'];
						var cardPrice=cardList['cardPrice'];
						var cardRare=cardList['cardRare'];
						var cardName=cardList['cardName'];
						var imgPath=cardList['imgPath'];
							console.log("卡號:"+cardNumber);
							console.log("單價:"+cardPrice);
							console.log("數量:"+cardTimes);
							console.log("更新時間:"+uppdate);

							createTableTree(cardNumber,cardPrice,cardTimes,cardRare,cardName,deckGameTitle,imgPath);
							totalNum=totalNum+parseInt(cardTimes,10);
							var tempEvaluate = math.evaluate(cardPrice*cardTimes);
							var tempTotalPrice = math.format(tempEvaluate, {precision: 14});
							totalPrice=Number(math.format(Number(totalPrice)+(Number(tempTotalPrice)), {precision: 14}));
							console.log("途中 total:"+totalPrice);	
								
					}
				console.log("total:"+totalPrice);
				console.log("卡片總數:"+totalNum);
				var dollarSign="¥";
				if(deckGameTitle === '12'){
					dollarSign="C$";
				}else{
					dollarSign="¥";
				}
				document.getElementById('deckTotalPrice').innerHTML=dollarSign+' '+moneyFormat(totalPrice.toString());
				var deckTotalPrice=document.getElementById('deckTotalPrice');
				deckTotalPrice.addEventListener("click",function(){
					var tableCardBody = document.getElementById("tableCardBody");
					var count = 0;
					if (tableCardBody.hasChildNodes()) {
						// 取得 foo 元素的所有子元素集合
						var children = tableCardBody.childNodes;

						// 可以用 for 迴圈來遍歷每一個子元素
						for (var i=0; i<children.length; ++i) {
							count=count+parseInt(children[i].childNodes[3].textContent);
						}
					}	
					Swal.fire(
					  '牌組資訊：',
					  '你的牌組有 '+count+' 張',
					  'question'
					)						
				});						
				document.getElementById('overlay-1').style.display='none';
				document.getElementById('overlay-2').style.display='none';	
				if((!totalNum===50)){alert('不足50張，有可能是資料抓取問題');}
			}
			
			var deckTable=document.getElementById('deckTable');		
			
			function createTableTree(cardNumber,cardPrice,cardTimes,cardRare,cardName,deckGameTitle,imgPath){
			var deckTablebody=document.getElementById('tableCardBody');		
				var tr = document.createElement("tr");							
				var tdCardNo = document.createElement("td");
				tdCardNo.setAttribute("class","tdClass");
				tdCardNo.setAttribute("id",cardNumber);
				tdCardNo.addEventListener("click", function(){
							console.log('cardNumber:'+cardNumber);
							
							//catch num
							var card_Num=cardNumber;
							if(card_Num.indexOf(' ')>=0){
								card_Num=card_Num.substr(0,card_Num.indexOf(' '));
							}else{
								card_Num=card_Num;				
							}

							var card_first=card_Num.substr(0,1);
							var card_second=card_Num.substr(0,card_Num.indexOf('-'));
								card_second=card_second.replace('/','_')
							var card_third=card_Num.replace('/','_');
								card_third=card_third.replace('-','_');								
							var urlCard="https://ws-tcg.com/wordpress/wp-content/cardimages/"+card_first.toLowerCase()+"/"+card_second.toLowerCase()+"/"+card_third.toLowerCase()+".png";				
							console.log('dec:'+deckGameTitle);
							if(deckGameTitle==='5'){
								urlCard="https://s3-ap-northeast-1.amazonaws.com/rebirth-fy.com/wordpress/wp-content/images/cardlist/"+card_third.toLowerCase()+".png";
							}else if(deckGameTitle==='12'){
								urlCard="https://en.ws-tcg.com/wp/wp-content/images/cardimages/"+card_third.toUpperCase()+".png";
							}
								
							
							var cardImg=document.getElementById('cardImg');
							cardImg.setAttribute("src",urlCard);
							var cardShowName=document.getElementById('cardName');
							cardShowName.innerHTML=cardName;
							var cardShowNumber=document.getElementById('exampleModalLongTitle');
							cardShowNumber.innerHTML=card_Num;
							$('#exampleModalCenter').modal('show');		
				});				

				
				var tdPrice = document.createElement("td");
				var tdTotalPrice = document.createElement("td");				
				var tdNum = document.createElement("td");
				var tdRare = document.createElement("td");
				tdCardNo.innerHTML=cardNumber;	
				tdPrice.innerHTML=cardPrice;	
				var tempEvaluate = math.evaluate(cardPrice*cardTimes);
				var tempTotalPrice = math.format(tempEvaluate, {precision: 14});
				tdTotalPrice.innerHTML=Number(tempTotalPrice);
				tdNum.innerHTML=cardTimes;	
				tdRare.innerHTML=cardRare;

				tr.appendChild(tdCardNo); 
				tr.appendChild(tdRare); 				
				tr.appendChild(tdPrice); 
				tr.appendChild(tdNum); 
				tr.appendChild(tdTotalPrice);
				deckTablebody.appendChild(tr);						
			}	

			var btncopyListener = document.getElementById("btn-copy");
			btncopyListener.addEventListener("click", function(){
				var deckCode=document.getElementById('deckCode').value;
				var deckSource=document.getElementById('deckSource').innerHTML;
				if(deckSource === 'bottleneko'){
						Swal.fire({
						  title: '將前往貓罐子網站，請問要繼續嗎?',
						  icon: 'warning',
						  showCancelButton: true,
						  confirmButtonColor: '#3085d6',
						  cancelButtonColor: '#d33',
						  confirmButtonText: '是的'
						}).then((result) => {
						  if (result.isConfirmed) {
							window.open("https://bottle-neko.web.app/deck/"+deckCode);
						  }
						})						
				}else if(deckSource === 'decklog'){
						Swal.fire({
						  title: '將前往DeckLog網站，請問要繼續嗎?',
						  icon: 'warning',
						  showCancelButton: true,
						  confirmButtonColor: '#3085d6',
						  cancelButtonColor: '#d33',
						  confirmButtonText: '是的'
						}).then((result) => {
						  if (result.isConfirmed) {
							window.open("https://decklog.bushiroad.com/view/"+deckCode);
						  }
						})
				}else if(deckSource === 'decklog-en'){
						Swal.fire({
						  title: '將前往DeckLog-EN網站，請問要繼續嗎?',
						  icon: 'warning',
						  showCancelButton: true,
						  confirmButtonColor: '#3085d6',
						  cancelButtonColor: '#d33',
						  confirmButtonText: '是的'
						}).then((result) => {
						  if (result.isConfirmed) {
							window.open("https://decklog-en.bushiroad.com/view/"+deckCode);
						  }
						})					
									
				}else if(deckSource === 'encoredecks'){
						Swal.fire({
						  title: '將前往EncoreDecks網站，請問要繼續嗎?',
						  icon: 'warning',
						  showCancelButton: true,
						  confirmButtonColor: '#3085d6',
						  cancelButtonColor: '#d33',
						  confirmButtonText: '是的'
						}).then((result) => {
						  if (result.isConfirmed) {
							window.open("https://www.encoredecks.com/deck/"+deckCode);
						  }
						})					
				}
			});	
				
			function removeTable(){
				var tabletree = document.getElementById('deckTable');
				tabletree.innerHTML="";
			}
			
		   function moneyFormat(str) {
				if (str.length <= 3 || str.indexOf(".") >= 0) {
					return str;
				}
				else {
					return moneyFormat(str.substr(0, str.length - 3)) + "," + (str.substr(str.length - 3));
				}
			} 
			
