#include "mosaic/agent.hpp"

#include "mosaic/features.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace mosaic {
namespace {

struct TokenSpan {
    std::string value;
    std::size_t begin = 0;
    std::size_t end = 0;
};

std::vector<TokenSpan> spans(const std::string& input) {
    std::vector<TokenSpan> result;
    std::size_t i = 0;
    while (i < input.size()) {
        const auto c = static_cast<unsigned char>(input[i]);
        const bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c >= 0x80;
        if (!word) {
            ++i;
            continue;
        }
        const std::size_t begin = i;
        std::string value;
        while (i < input.size()) {
            const auto d = static_cast<unsigned char>(input[i]);
            const bool in_word = (d >= 'a' && d <= 'z') || (d >= 'A' && d <= 'Z') ||
                                 (d >= '0' && d <= '9') || d >= 0x80;
            if (!in_word) break;
            char ch = input[i];
            const auto uch = static_cast<unsigned char>(ch);
            if (uch >= 'A' && uch <= 'Z') ch = static_cast<char>(uch - 'A' + 'a');
            value.push_back(ch);
            ++i;
        }
        result.push_back({std::move(value), begin, i});
    }
    return result;
}

bool has_token(const std::vector<TokenSpan>& tokens, const std::string& value) {
    return std::any_of(tokens.begin(), tokens.end(),
                       [&](const TokenSpan& token) { return token.value == value; });
}

std::optional<std::size_t> find_sequence(
    const std::vector<TokenSpan>& tokens, const std::vector<std::string>& pattern) {
    if (pattern.empty() || tokens.size() < pattern.size()) return std::nullopt;
    for (std::size_t i = 0; i + pattern.size() <= tokens.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < pattern.size(); ++j) {
            if (tokens[i + j].value != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) return i;
    }
    return std::nullopt;
}

std::string capture_after(const std::string& input,
                          const std::vector<TokenSpan>& tokens,
                          std::size_t token_after, std::size_t max_words,
                          bool stop_at_and) {
    if (token_after >= tokens.size()) return {};
    std::size_t last = token_after;
    std::size_t count = 0;
    for (std::size_t i = token_after; i < tokens.size() && count < max_words; ++i) {
        if (stop_at_and && i > token_after && tokens[i].value == "and") break;
        last = i;
        ++count;
    }
    std::string value = input.substr(tokens[token_after].begin,
                                     tokens[last].end - tokens[token_after].begin);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.erase(value.begin());
    while (!value.empty()) {
        const char c = value.back();
        if (std::isspace(static_cast<unsigned char>(c)) || c == '.' || c == ',' ||
            c == '!' || c == '?' || c == ';' || c == ':') {
            value.pop_back();
        } else {
            break;
        }
    }
    return value;
}

bool explicit_correction(const std::vector<TokenSpan>& tokens) {
    if (tokens.empty()) return false;
    return tokens.front().value == "actually" ||
           tokens.front().value == "correction" ||
           tokens.front().value == "wait" ||
           tokens.front().value == "no" ||
           has_token(tokens, "instead") ||
           has_token(tokens, "correcting") ||
           (has_token(tokens, "changed") && has_token(tokens, "mind")) ||
           (has_token(tokens, "i") && has_token(tokens, "meant"));
}

std::optional<std::string> capture_name(
    const std::string& input, const std::vector<TokenSpan>& tokens) {
    const auto position = find_sequence(tokens, {"my", "name", "is"});
    if (position)
        return capture_after(input, tokens, *position + 3, 2, true);

    const auto called = find_sequence(tokens, {"i", "am", "called"});
    if (called)
        return capture_after(input, tokens, *called + 3, 2, true);
    const auto im_called = find_sequence(tokens, {"i", "m", "called"});
    if (im_called)
        return capture_after(input, tokens, *im_called + 3, 2, true);
    return std::nullopt;
}

std::optional<std::string> capture_project(
    const std::string& input, const std::vector<TokenSpan>& tokens) {
    const std::vector<std::vector<std::string>> patterns = {
        {"i", "am", "building"}, {"i", "m", "building"},
        {"i", "am", "making"}, {"i", "m", "making"},
        {"i", "am", "developing"}, {"i", "m", "developing"},
        {"i", "am", "working", "on"}, {"i", "m", "working", "on"},
        {"my", "project", "is"}
    };
    std::optional<std::pair<std::size_t, std::string>> best;
    for (const auto& pattern : patterns) {
        const auto position = find_sequence(tokens, pattern);
        if (!position) continue;
        const auto value = capture_after(input, tokens, *position + pattern.size(), 12, true);
        if (!value.empty() && (!best || *position < best->first))
            best = std::make_pair(*position, value);
    }
    if (!best) return std::nullopt;
    return best->second;
}

std::optional<std::pair<std::string, std::string>> capture_profile_fact(
    const std::string& input, const std::vector<TokenSpan>& tokens) {
    const std::vector<std::pair<std::vector<std::string>, std::string>> patterns = {
        {{"i", "don", "t", "like"}, "dislikes"},
        {{"i", "do", "not", "like"}, "dislikes"},
        {{"i", "dislike"}, "dislikes"},
        {{"i", "hate"}, "dislikes"},
        {{"i", "live", "in"}, "location"},
        {{"i", "am", "from"}, "origin"},
        {{"i", "m", "from"}, "origin"},
        {{"i", "work", "as"}, "occupation"},
        {{"i", "am", "studying"}, "study_subject"},
        {{"i", "m", "studying"}, "study_subject"},
        {{"i", "study"}, "study_subject"},
        {{"i", "prefer"}, "preference"},
        {{"i", "enjoy"}, "preference"},
        {{"i", "love"}, "preference"},
        {{"i", "like"}, "preference"},
        {{"my", "hobby", "is"}, "hobby"},
        {{"my", "goal", "is"}, "goal"}
    };

    for (const auto& item : patterns) {
        const auto position = find_sequence(tokens, item.first);
        if (!position) continue;
        const std::size_t after = *position + item.first.size();
        const auto value = capture_after(input, tokens, after, 8, true);
        if (!value.empty()) return std::make_pair(item.second, value);
    }

    for (const auto& adjective : {std::string("favorite"), std::string("favourite")}) {
        const auto position = find_sequence(tokens, {"my", adjective});
        if (!position) continue;
        std::size_t cursor = *position + 2;
        std::string category;
        std::size_t category_words = 0;
        while (cursor < tokens.size() && tokens[cursor].value != "is" &&
               category_words < 3) {
            if (!category.empty()) category.push_back('_');
            category += tokens[cursor].value;
            ++cursor;
            ++category_words;
        }
        if (category.empty() || cursor >= tokens.size() || tokens[cursor].value != "is")
            continue;
        const auto value = capture_after(input, tokens, cursor + 1, 8, true);
        if (!value.empty())
            return std::make_pair("favorite_" + category, value);
    }
    return std::nullopt;
}

std::optional<std::string> profile_query_predicate(
    const std::vector<TokenSpan>& tokens) {
    if (tokens.empty()) return std::nullopt;
    for (const auto& adjective : {std::string("favorite"), std::string("favourite")}) {
        const auto position = find_sequence(tokens, {"my", adjective});
        if (position) {
            std::string category;
            for (std::size_t i = *position + 2; i < tokens.size() && i < *position + 5; ++i) {
                if (!category.empty()) category.push_back('_');
                category += tokens[i].value;
            }
            return category.empty()
                ? std::optional<std::string>("favorite")
                : std::optional<std::string>("favorite_" + category);
        }
    }
    if (has_token(tokens, "where") && has_token(tokens, "live")) return "location";
    if (has_token(tokens, "where") && has_token(tokens, "from")) return "origin";
    if (has_token(tokens, "what") && has_token(tokens, "like")) return "preference";
    if (has_token(tokens, "what") && has_token(tokens, "prefer")) return "preference";
    if (has_token(tokens, "what") && has_token(tokens, "hobby")) return "hobby";
    if (has_token(tokens, "what") && has_token(tokens, "goal")) return "goal";
    if (has_token(tokens, "what") &&
        (has_token(tokens, "job") || has_token(tokens, "occupation") ||
         has_token(tokens, "work"))) return "occupation";
    if (has_token(tokens, "what") &&
        (has_token(tokens, "study") || has_token(tokens, "studying"))) return "study_subject";
    if (has_token(tokens, "what") && has_token(tokens, "dislike")) return "dislikes";
    return std::nullopt;
}

std::string profile_confirmation(const std::string& predicate,
                                 const std::string& value,
                                 bool correction) {
    const std::string prefix = correction ? "Understood. " : "Got it. ";
    if (predicate == "location")
        return prefix + "I'll remember that you live in " + value + ".";
    if (predicate == "origin")
        return prefix + "I'll remember that you're from " + value + ".";
    if (predicate == "occupation")
        return prefix + "I'll remember that you work as " + value + ".";
    if (predicate == "preference")
        return prefix + "I'll remember that you like " + value + ".";
    if (predicate == "dislikes")
        return prefix + "I'll remember that you dislike " + value + ".";
    if (predicate == "hobby")
        return prefix + "I'll remember that your hobby is " + value + ".";
    if (predicate == "goal")
        return prefix + "I'll remember that your goal is " + value + ".";
    if (predicate == "study_subject")
        return prefix + "I'll remember that you're studying " + value + ".";
    if (predicate.rfind("favorite_", 0) == 0) {
        std::string category = predicate.substr(9);
        std::replace(category.begin(), category.end(), '_', ' ');
        return prefix + "I'll remember that your favorite " + category + " is " + value + ".";
    }
    return prefix + "I'll remember that detail.";
}

std::string profile_answer(const std::string& predicate, const std::string& value) {
    if (predicate == "location") return "You told me you live in " + value + ".";
    if (predicate == "origin") return "You told me you're from " + value + ".";
    if (predicate == "occupation") return "You told me you work as " + value + ".";
    if (predicate == "preference") return "You told me you like " + value + ".";
    if (predicate == "dislikes") return "You told me you dislike " + value + ".";
    if (predicate == "hobby") return "You told me your hobby is " + value + ".";
    if (predicate == "goal") return "You told me your goal is " + value + ".";
    if (predicate == "study_subject") return "You told me you're studying " + value + ".";
    if (predicate.rfind("favorite_", 0) == 0) {
        std::string category = predicate.substr(9);
        std::replace(category.begin(), category.end(), '_', ' ');
        return "You told me your favorite " + category + " is " + value + ".";
    }
    return "You haven't told me that yet.";
}

bool is_greeting(const std::vector<TokenSpan>& tokens) {
    if (tokens.empty()) return false;
    const auto& first = tokens.front().value;
    if (first == "hi" || first == "hello" || first == "hey" ||
        first == "howdy" || first == "yo") return true;
    if (tokens.size() >= 2 && first == "good" &&
        (tokens[1].value == "morning" || tokens[1].value == "afternoon" ||
         tokens[1].value == "evening")) return true;
    return false;
}

bool is_thanks(const std::vector<TokenSpan>& tokens) {
    return has_token(tokens, "thanks") ||
           find_sequence(tokens, {"thank", "you"}).has_value();
}

bool is_farewell(const std::vector<TokenSpan>& tokens) {
    if (tokens.empty()) return false;
    const auto& first = tokens.front().value;
    return first == "bye" || first == "goodbye" || first == "farewell" ||
           (first == "see" && tokens.size() > 1 && tokens[1].value == "you");
}

bool is_name_question(const std::vector<TokenSpan>& tokens) {
    return has_token(tokens, "name") &&
           (has_token(tokens, "my") || has_token(tokens, "remember") ||
            (has_token(tokens, "tell") && has_token(tokens, "me")));
}

bool is_project_question(const std::vector<TokenSpan>& tokens) {
    const bool asks = has_token(tokens, "what") || has_token(tokens, "which") ||
                      has_token(tokens, "remember") || has_token(tokens, "tell");
    const bool subject = has_token(tokens, "project") || has_token(tokens, "building") ||
                         has_token(tokens, "making") || has_token(tokens, "developing") ||
                         has_token(tokens, "working");
    return asks && subject && !has_token(tokens, "your");
}

struct LexicalQuery {
    std::string term;
    bool synonyms = false;
};

std::optional<LexicalQuery> lexical_query(const std::vector<TokenSpan>& tokens) {
    if (tokens.empty()) return std::nullopt;
    std::size_t begin = tokens.size();
    std::size_t end = tokens.size();
    bool synonyms = false;

    if (tokens.front().value == "define" && tokens.size() > 1) {
        begin = 1;
    } else if (tokens.front().value == "meaning" && tokens.size() > 2 &&
               tokens[1].value == "of") {
        begin = 2;
    } else if (tokens.front().value == "what" && tokens.size() > 3 &&
               tokens[1].value == "does" && tokens.back().value == "mean") {
        begin = 2;
        end = tokens.size() - 1;
    } else if (tokens.front().value == "what" && tokens.size() > 2 &&
               tokens[1].value == "is") {
        if (tokens.size() > 5 && tokens[2].value == "the" &&
            tokens[3].value == "definition" && tokens[4].value == "of") {
            begin = 5;
        } else if (tokens.size() > 5 && tokens[2].value == "the" &&
                   tokens[3].value == "meaning" && tokens[4].value == "of") {
            begin = 5;
        } else {
            begin = 2;
        }
    } else {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if ((tokens[i].value == "synonyms" || tokens[i].value == "synonym") &&
                i + 1 < tokens.size()) {
                std::size_t separator = i + 1;
                while (separator < tokens.size() &&
                       tokens[separator].value != "for" &&
                       tokens[separator].value != "of") ++separator;
                if (separator + 1 < tokens.size()) {
                    begin = separator + 1;
                    synonyms = true;
                }
                break;
            }
        }
    }

    if (begin >= tokens.size()) return std::nullopt;
    if (end == tokens.size()) end = tokens.size();
    if (begin < end && (tokens[begin].value == "a" ||
                        tokens[begin].value == "an" ||
                        tokens[begin].value == "the")) ++begin;
    while (end > begin && (tokens[end - 1].value == "please" ||
                           tokens[end - 1].value == "thanks")) --end;
    if (end <= begin || end - begin > 5) return std::nullopt;

    std::string term;
    for (std::size_t i = begin; i < end; ++i) {
        if (!term.empty()) term.push_back(' ');
        term += tokens[i].value;
    }
    if (term.empty()) return std::nullopt;
    return LexicalQuery{std::move(term), synonyms};
}

std::string first_related_items(const std::string& list, std::size_t limit = 5) {
    std::string result;
    std::size_t begin = 0;
    std::size_t count = 0;
    while (begin < list.size() && count < limit) {
        const auto end = list.find(';', begin);
        const auto item = list.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin);
        if (!item.empty()) {
            if (!result.empty()) result += ", ";
            result += item;
            ++count;
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

class ExpressionParser {
public:
    explicit ExpressionParser(const std::string& expression) : text_(expression) {}

    std::optional<double> parse() {
        position_ = 0;
        const auto value = parse_expression();
        skip_spaces();
        if (!value || position_ != text_.size() || !std::isfinite(*value))
            return std::nullopt;
        return value;
    }

private:
    void skip_spaces() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
    }

    std::optional<double> parse_expression() {
        auto lhs = parse_term();
        if (!lhs) return std::nullopt;
        while (true) {
            skip_spaces();
            if (position_ >= text_.size() ||
                (text_[position_] != '+' && text_[position_] != '-')) return lhs;
            const char op = text_[position_++];
            const auto rhs = parse_term();
            if (!rhs) return std::nullopt;
            *lhs = op == '+' ? *lhs + *rhs : *lhs - *rhs;
        }
    }

    std::optional<double> parse_term() {
        auto lhs = parse_factor();
        if (!lhs) return std::nullopt;
        while (true) {
            skip_spaces();
            if (position_ >= text_.size() ||
                (text_[position_] != '*' && text_[position_] != '/')) return lhs;
            const char op = text_[position_++];
            const auto rhs = parse_factor();
            if (!rhs || (op == '/' && std::abs(*rhs) < 1e-15)) return std::nullopt;
            *lhs = op == '*' ? *lhs * *rhs : *lhs / *rhs;
        }
    }

    std::optional<double> parse_factor() {
        skip_spaces();
        if (position_ >= text_.size()) return std::nullopt;
        if (text_[position_] == '+' || text_[position_] == '-') {
            const char sign = text_[position_++];
            auto value = parse_factor();
            if (value && sign == '-') *value = -*value;
            return value;
        }
        if (text_[position_] == '(') {
            ++position_;
            auto value = parse_expression();
            skip_spaces();
            if (!value || position_ >= text_.size() || text_[position_] != ')')
                return std::nullopt;
            ++position_;
            return value;
        }
        const char* begin = text_.c_str() + position_;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin || !std::isfinite(value)) return std::nullopt;
        position_ += static_cast<std::size_t>(end - begin);
        return value;
    }

    const std::string& text_;
    std::size_t position_ = 0;
};

void replace_all(std::string& text, const std::string& from, const std::string& to) {
    std::size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
}

std::optional<double> evaluate_arithmetic(const std::string& input) {
    std::string expression = mosaic_lower_ascii(input);
    const bool has_operator_word =
        expression.find(" plus ") != std::string::npos ||
        expression.find(" minus ") != std::string::npos ||
        expression.find(" times ") != std::string::npos ||
        expression.find(" multiplied by ") != std::string::npos ||
        expression.find(" divided by ") != std::string::npos ||
        expression.find(" over ") != std::string::npos;
    replace_all(expression, "multiplied by", "*");
    replace_all(expression, "divided by", "/");
    replace_all(expression, " plus ", "+");
    replace_all(expression, " minus ", "-");
    replace_all(expression, " times ", "*");
    replace_all(expression, " over ", "/");
    replace_all(expression, "what is", " ");
    replace_all(expression, "what's", " ");
    replace_all(expression, "calculate", " ");
    replace_all(expression, "compute", " ");
    replace_all(expression, "solve", " ");
    replace_all(expression, "how much is", " ");

    while (!expression.empty() &&
           (expression.back() == '?' || expression.back() == '!' ||
            expression.back() == ',' || expression.back() == ';' ||
            expression.back() == ':')) expression.pop_back();

    const bool has_digit = std::any_of(expression.begin(), expression.end(),
        [](unsigned char c) { return std::isdigit(c) != 0; });
    const bool has_operator = has_operator_word ||
        expression.find_first_of("+-*/()") != std::string::npos;
    if (!has_digit || !has_operator) return std::nullopt;
    ExpressionParser parser(expression);
    return parser.parse();
}

std::string format_number(double value) {
    if (std::abs(value) < 1e-12) value = 0.0;
    std::ostringstream output;
    output << std::setprecision(12) << std::defaultfloat << value;
    return output.str();
}

std::string hex_encode(const std::string& value) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() * 2);
    for (const unsigned char c : value) {
        out.push_back(digits[c >> 4]);
        out.push_back(digits[c & 15]);
    }
    return out;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::optional<std::string> hex_decode(const std::string& value) {
    if (value.size() % 2 != 0) return std::nullopt;
    std::string out;
    out.reserve(value.size() / 2);
    for (std::size_t i = 0; i < value.size(); i += 2) {
        const int a = hex_value(value[i]);
        const int b = hex_value(value[i + 1]);
        if (a < 0 || b < 0) return std::nullopt;
        out.push_back(static_cast<char>((a << 4) | b));
    }
    return out;
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const auto at = line.find('\t', start);
        if (at == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, at - start));
        start = at + 1;
    }
    return fields;
}

}  // namespace

bool Agent::load_intent_model(const std::string& path, std::string* error) {
    return intent_model_.load(path, error);
}

bool Agent::load_lexicon(const std::string& path, std::string* error) {
    return lexicon_.load(path, error);
}

void Agent::remember(const std::string& predicate, const std::string& value,
                     bool correction) {
    if (value.empty()) return;
    const auto active = state_.query("user", predicate);
    for (const auto& fact : active) {
        if (fact.provenance == "user" && fact.value == value) return;
    }

    Delta delta;
    delta.opcode = correction && !active.empty()
        ? OpCode::CorrectFact : OpCode::AssertFact;
    delta.subject = "user";
    delta.predicate = predicate;
    delta.value = value;
    delta.provenance = "user";
    delta.note = correction ? "explicit user correction" : "user statement";
    delta.turn = turn_;
    if (delta.opcode == OpCode::CorrectFact) delta.fact_id = active.front().id;
    (void)state_.apply(delta);
}

std::string Agent::respond(const std::string& message) {
    ++turn_;
    const auto tokens = spans(message);
    if (tokens.empty()) return "I didn't catch a message. What would you like to talk about?";

    const bool correction = explicit_correction(tokens);

    if (const auto name = capture_name(message, tokens)) {
        remember("name", *name, correction);
        return correction
            ? "Got it. I'll use " + *name + " as your name."
            : "Nice to meet you, " + *name + ".";
    }

    if (const auto project = capture_project(message, tokens)) {
        remember("current_project", *project, correction);
        return correction
            ? "Understood. Your current project is " + *project + "."
            : "That sounds interesting. You're working on " + *project + ".";
    }

    if (const auto profile = capture_profile_fact(message, tokens)) {
        remember(profile->first, profile->second, correction);
        return profile_confirmation(profile->first, profile->second, correction);
    }

    if (has_token(tokens, "name") && has_token(tokens, "your")) {
        return "I'm MCR, a compact conversational system built around explicit meaning, memory and reasoning.";
    }

    if (is_name_question(tokens)) {
        const auto facts = state_.query("user", "name");
        if (!facts.empty())
            return "You told me your name is " + facts.front().value + ".";
        return "You haven't told me your name yet. What should I call you?";
    }

    if (is_project_question(tokens)) {
        const auto facts = state_.query("user", "current_project");
        if (!facts.empty())
            return "Your current project is " + facts.front().value + ".";
        return "You haven't told me what you're building yet. What is the project?";
    }

    if (const auto predicate = profile_query_predicate(tokens)) {
        const auto facts = state_.query("user", *predicate);
        if (!facts.empty()) return profile_answer(*predicate, facts.front().value);
        return "You haven't told me that yet.";
    }

    if (is_greeting(tokens)) {
        return "Hi! Good to see you. What's on your mind?";
    }

    if (is_thanks(tokens)) {
        return "You're welcome. What would you like to work on next?";
    }

    if (is_farewell(tokens)) {
        return "Goodbye! I'll be here when you want to continue.";
    }

    if (has_token(tokens, "how") && has_token(tokens, "you") &&
        (has_token(tokens, "are") || has_token(tokens, "doing"))) {
        return "I'm running and ready to help. What are you thinking about?";
    }

    if (has_token(tokens, "who") && has_token(tokens, "you")) {
        return "I'm MCR, a compact conversational system built around explicit meaning, memory and reasoning.";
    }

    if (has_token(tokens, "what") && has_token(tokens, "can") &&
        (has_token(tokens, "do") || has_token(tokens, "ask"))) {
        return "You can tell me facts to remember, correct earlier details, ask questions about our conversation, or give me arithmetic to solve. My broader knowledge and language skills are still limited.";
    }

    if (lexicon_.loaded()) {
        if (const auto query = lexical_query(tokens)) {
            const auto senses = lexicon_.lookup(query->term, 3);
            if (!senses.empty()) {
                const auto& sense = *senses.front();
                if (query->synonyms) {
                    const auto synonyms = first_related_items(sense.synonyms);
                    if (!synonyms.empty())
                        return "Some synonyms for " + query->term + " are " + synonyms + ".";
                    return "I found " + query->term + " in my dictionary, but this sense has no listed synonyms.";
                }
                std::string answer = "In Open English WordNet, " + query->term +
                    " (" + sense.pos + ") means: " + sense.definition;
                if (!sense.synonyms.empty())
                    answer += " Related words: " + first_related_items(sense.synonyms) + ".";
                if (!sense.hypernyms.empty())
                    answer += " A broader category is: " + first_related_items(sense.hypernyms, 3) + ".";
                if (senses.size() > 1)
                    answer += " The word also has other recorded senses.";
                return answer;
            }
            return "I don't have a local WordNet entry for " + query->term +
                ". I won't invent a definition.";
        }
    }

    if (const auto result = evaluate_arithmetic(message)) {
        return "The result is " + format_number(*result) + ".";
    }

    const IntentPrediction prediction = intent_model_.predict(message);
    if (prediction.probability >= 0.42F) {
        if (prediction.label == "greeting")
            return "Hi! What's on your mind?";
        if (prediction.label == "goodbye")
            return "Goodbye! Talk to you again soon.";
        if (prediction.label == "thank_you")
            return "You're welcome!";
        if (prediction.label == "what_is_your_name" ||
            prediction.label == "who_made_you" ||
            prediction.label == "are_you_a_bot")
            return "I'm MCR, a compact conversational system. I don't use a pretrained language model.";
        if (prediction.label == "what_can_i_ask_you")
            return "Ask about details you've told me, correct remembered facts, or try arithmetic. Other open-domain questions may exceed what I know.";
        if (prediction.label == "calculator")
            return "Send me the calculation using numbers and +, -, *, /, or parentheses.";
        if (prediction.label == "tell_joke")
            return "Why did the programmer confuse Halloween and Christmas? Because OCT 31 equals DEC 25.";
    }

    if (!message.empty() && message.back() == '?') {
        return "I don't have enough grounded information to answer that reliably yet. Can you give me a little more context?";
    }
    return "I follow. Tell me a little more about what you're thinking, and I'll work through it with you.";
}

void Agent::reset() {
    state_ = StateKernel{};
    turn_ = 0;
}

bool Agent::save_memory(const std::string& path, std::string* error) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        if (error) *error = "cannot write memory file: " + path;
        return false;
    }
    output << "MCR_MEMORY_V1\n";
    for (const auto& fact : state_.fact_log()) {
        output << (fact.active ? 1 : 0) << '\t'
               << fact.turn << '\t'
               << static_cast<unsigned int>(fact.epistemic) << '\t'
               << fact.confidence_q16 << '\t'
               << hex_encode(fact.subject) << '\t'
               << hex_encode(fact.predicate) << '\t'
               << hex_encode(fact.value) << '\t'
               << hex_encode(fact.provenance) << '\t'
               << hex_encode(fact.note) << '\n';
    }
    if (!output) {
        if (error) *error = "failed while writing memory file";
        return false;
    }
    return true;
}

bool Agent::load_memory(const std::string& path, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return true;  // Missing memory is a normal first-run condition.

    std::string line;
    if (!std::getline(input, line) || line != "MCR_MEMORY_V1") {
        if (error) *error = "unsupported or corrupt memory file";
        return false;
    }

    StateKernel loaded;
    std::uint64_t max_turn = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_tabs(line);
        if (fields.size() != 9) {
            if (error) *error = "invalid memory record";
            return false;
        }
        const auto subject = hex_decode(fields[4]);
        const auto predicate = hex_decode(fields[5]);
        const auto value = hex_decode(fields[6]);
        const auto provenance = hex_decode(fields[7]);
        const auto note = hex_decode(fields[8]);
        if (!subject || !predicate || !value || !provenance || !note) {
            if (error) *error = "invalid hex data in memory record";
            return false;
        }
        try {
            const auto turn = static_cast<std::uint64_t>(std::stoull(fields[1]));
            const auto epistemic = static_cast<unsigned int>(std::stoul(fields[2]));
            const auto confidence = static_cast<std::uint32_t>(std::stoul(fields[3]));
            if (epistemic > static_cast<unsigned int>(EpistemicStatus::Denied) ||
                confidence > 65535U || (fields[0] != "0" && fields[0] != "1")) {
                if (error) *error = "memory record value out of range";
                return false;
            }
            Delta delta;
            delta.opcode = OpCode::AssertFact;
            delta.subject = *subject;
            delta.predicate = *predicate;
            delta.value = *value;
            delta.provenance = *provenance;
            delta.note = *note;
            delta.turn = turn;
            delta.epistemic = static_cast<EpistemicStatus>(epistemic);
            delta.confidence_q16 = confidence;
            const auto result = loaded.apply(delta);
            if (!result.ok) {
                if (error) *error = "could not restore fact: " + result.message;
                return false;
            }
            if (fields[0] == "0") {
                Delta retract;
                retract.opcode = OpCode::RetractFact;
                retract.fact_id = result.created_id;
                retract.note = "restored historical inactive record";
                if (!loaded.apply(retract).ok) {
                    if (error) *error = "could not restore inactive fact";
                    return false;
                }
            }
            max_turn = std::max(max_turn, turn);
        } catch (const std::exception&) {
            if (error) *error = "invalid numeric value in memory record";
            return false;
        }
    }
    state_ = std::move(loaded);
    turn_ = max_turn;
    return true;
}

}  // namespace mosaic
