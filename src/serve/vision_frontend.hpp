#pragma once
// The shared vision frontend (docs/vision.md): the checkpoint's own Jinja
// template still renders roles, tools and reasoning, and each user image_url
// part is replaced by its trained delimiters plus one pad token per visual
// token before rendering. Every family needs the same three things, so the
// scan and the span bookkeeping live here: the delimiter ids, the text those
// delimiters tokenize to, and the processor geometry.
#include <string>

#include "serve/frontend.hpp"
#include "serve/image_inputs.hpp"

namespace dgpp::serve {
struct VisionSpec {
  ImageTokens ids;
  std::string start_text, pad_text, end_text;
  ImagePreprocess preprocess;
};

class VisionFrontend : public TextFrontend {
 public:
  VisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                 VisionSpec spec);
  bool supports_images() const override { return true; }
  ChatInput prepare_chat(const minijson::Value& globals) const override;
  const VisionSpec& spec() const { return spec_; }

 private:
  VisionSpec spec_;
};

// GLM-5.3-Flash: the 28-pixel canvas with black padding and GLM's trained
// <|begin_of_image|> / <|image|> / <|end_of_image|> delimiters.
VisionSpec glm_vision_spec(ImageTokens ids);
// Qwen3.8-Flash-Next: the 32-pixel canvas, rounded to the nearest multiple of
// the grid and never padded, delimited by the checkpoint's vision_start,
// image_token and vision_end ids. Their surface forms come from the
// tokenizer's vocabulary rather than a literal, so the ids stay authoritative;
// the frontend constructor still checks that each one round-trips to exactly
// its id.
VisionSpec qwen_vision_spec(dgpp::ImageTokens ids, const dgpp::text::Tokenizer& tok);

}  // namespace dgpp::serve
