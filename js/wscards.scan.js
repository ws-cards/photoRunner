	window.onOpenCvReady = function () { 
		//swal("載入成功！", "可以開始使用!","success")
		//rundo();
	}

	function rundo(){
		//var src = cv.imread('src-image');
	}
	
	$('.nav-link').PushMenu('collapse');	
			var annButton=document.getElementById('annButton');	
			annButton.addEventListener("click", function(){
				$('#announceexampleModalCenter').modal('show');	
	});	
	var requestURL = 'https://divine-vehicle-292507.dt.r.appspot.com/api/getImage.jsp?file=';	
	var requestDeckLog = new XMLHttpRequest();	
	function sendImg(){
        var fd = new FormData();
		var files = $('#imgInp')[0].files;//image
        if(files.length > 0 ){
           fd.append('file',files[0]);
           $.ajax({
              url: requestURL,
              type: 'get',
              data: fd,
              contentType: false,
              processData: false,
              success: function(response){
			  console.log(response);
                 if(response != 0){
                    $("#img").attr("src",response); 
                    $(".preview img").show(); // Display image element
                 }else{
                    alert('file not uploaded');
                 }
              },
           });
        }else{
           alert("Please select a file.");
        }		

	}
