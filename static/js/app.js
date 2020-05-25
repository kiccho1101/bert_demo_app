var data = [];
var token = "";

jQuery(document).ready(function () {
  let sentences = [
    "I am eating a piece of bread.",
    "He is eating pizza.",
    "I am lying on the bed.",
    "She is sleeping.",
    "Bobby was arrested.",
    "Bobby was unhappy.",
  ];
  $("#vec_input").val(sentences.join("\n"));

  // Vectorize Sentence
  $("#vectorize_single_sentence").click(function (e) {
    const input_text = $("#vec_sin_input").val();
    if (input_text !== "") {
      $("#vec_sin_code").text(input_text);
      $.ajax({
        url: "/vectorize_sentences",
        type: "post",
        contentType: "application/json",
        dataType: "json",
        data: JSON.stringify({
          input_text: input_text,
        }),
        beforeSend: function () {
          $(".overlay").show();
        },
        complete: function () {
          $(".overlay").hide();
        },
      })
        .done(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
          $("#vec_sin_result_bert").val(
            `[\n    ${jsondata["bert"][0]
              .map(function (a) {
                return String(a);
              })
              .join(",\n    ")}\n]`
          );
        })
        .fail(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
        });
    }
  });

  // Vectorize Sentence
  $("#vectorize_sentences").click(function (e) {
    const input_text = $("#vec_input").val();
    if (input_text.length > 0) {
      const text = input_text.split("\n");
      $.ajax({
        url: "/vectorize_sentences",
        type: "post",
        contentType: "application/json",
        dataType: "json",
        data: JSON.stringify({
          input_text: input_text,
        }),
      })
        .done(function (jsondata, textStatus, jqXHR) {
          $.ajax({
            url: "/umap_comp",
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
              data: jsondata["bert"],
              n_neighbors: 2,
              min_dist: 0.01,
            }),
            beforeSend: function () {
              $(".overlay").show();
            },
            complete: function () {
              $(".overlay").hide();
            },
          })
            .done(function (compdata, textStatus, jqXHR) {
              $("#vec_plot").css("height", "400px");
              console.log(compdata);
              const x = [];
              const y = [];
              for (const xy of compdata) {
                x.push(xy[0]);
                y.push(xy[1]);
              }
              const data = [
                {
                  x: x,
                  y: y,
                  text: text,
                  type: "scatter",
                  mode: "markers",
                },
              ];
              const layout = {
                hovermode: "closest",
                title: "Plot the vectorized sentences",
              };
              Plotly.newPlot("vec_plot", data, layout);
            })
            .fail(function (jsondata2, textStatus, jqXHR) {
              console.log(jsondata2);
            });
        })
        .fail(function (jsondata1, textStatus, jqXHR) {
          console.log(jsondata1);
        });
    }
  });

  // Sentiment Analysis
  $("#sentiment_analysis").click(function (e) {
    const input_text = $("#sa_input").val();
    if (input_text !== "") {
      // $("#vec_sin_code").text(input_text);
      $.ajax({
        url: "/sentiment_analysis",
        type: "post",
        contentType: "application/json",
        dataType: "json",
        data: JSON.stringify({
          input_text: input_text,
        }),
        beforeSend: function () {
          $(".overlay").show();
        },
        complete: function () {
          $(".overlay").hide();
        },
      })
        .done(function (score, textStatus, jqXHR) {
          let emoji = "";
          if (score > 0.7) {
            emoji = "😋";
          } else if (score < 0.3) {
            emoji = "😫";
          } else {
            emoji = "😃";
          }
          $("#sa_result_emoji").text(emoji);
          $("#sa_result_score").text(String(score));
        })
        .fail(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
        });
    }
  });

  // Next Word Prediction
  $("#nwp_input").on("keyup", function (e) {
    if (e.key == " ") {
      $.ajax({
        url: "/get_end_predictions",
        type: "post",
        contentType: "application/json",
        dataType: "json",
        data: JSON.stringify({
          input_text: $("#nwp_input").val(),
          top_k: 5,
        }),
        beforeSend: function () {
          $(".overlay").show();
        },
        complete: function () {
          $(".overlay").hide();
        },
      })
        .done(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
          $("#nwp_result_bert").val(jsondata["bert"]);
        })
        .fail(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
        });
    }
  });
});
