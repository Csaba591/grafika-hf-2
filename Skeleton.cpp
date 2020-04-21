

//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Mészáros Csaba Máté
// Neptun : UY8Q7D
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *const vertexSource =
        R"(
    #version 330				// Shader 3.3
    precision highp float;		// normal floats, makes no difference on desktop computers

    layout(location = 0) in vec2 cVertexPosition;
    out vec2 texcoord;

    void main() {
        texcoord = (cVertexPosition + vec2(1, 1)) / 2;
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);		// transform vp from modeling space to normalized device space
    }
)";

// fragment shader in GLSL
const char *const fragmentSource =
        R"(
    #version 330			// Shader 3.3
    precision highp float;	// normal floats, makes no difference on desktop computers

    uniform sampler2D textureUnit;
    in vec2 texcoord;
    out vec4 fragmentColor;

    void main() {
        fragmentColor = texture(textureUnit, texcoord);
    }
)";

GPUProgram gpuProgram; // vertex and fragment shaders

enum MaterialType { ROUGH, REFLECTIVE };
struct Material {
    vec3 ka, kd, ks;
    float shininess;
    vec3 F0;
    MaterialType type;
    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * M_PI;
        kd = _kd; ks = _ks;
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        vec3 one(1, 1, 1);
        F0 = ((n - one)*(n - one) + kappa*kappa) / ((n + one)*(n + one) + kappa*kappa);
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material * material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) {
        start = _start;
        dir = normalize(_dir);
    }
};

class Quadratics {
    mat4 Q;
public:
    Quadratics(vec4 r1, vec4 r2, vec4 r3, vec4 r4) {
        Q = mat4(r1.x, r1.y, r1.z, r1.w,
                 r2.x, r2.y, r2.z, r2.w,
                 r3.x, r3.y, r3.z, r3.w,
                 r4.x, r4.y, r4.z, r4.w);
    }
    float f(vec4 x) {
        return dot(x * Q, x);
    }
    float fv(vec4 r, vec4 p) {
        return dot(r * Q, p);
    }
    vec3 gradf(vec4 x) {
        vec4 g = x * Q * 2;
        return vec3(g.x, g.y, g.z);
    }
};

class Intersectable {
protected:
    Material * material;
public:
    virtual Hit intersect(const Ray & ray) = 0;
};

class Paraboloid : public Intersectable, public Quadratics {
    vec3 orig;
    float height;
public:
    Paraboloid(vec3 t, float xw, float zw, float _height, Material* _material)
            : Quadratics(
            vec4(   1/(xw*xw),     0,            0, -t.x/(xw*xw)),
            vec4(           0,     0,            0,        -0.5f),
            vec4(           0,     0,    1/(zw*zw), -t.z/(zw*zw)),
            vec4(-t.x/(xw*xw), -0.5f, -t.z/(zw*zw), t.y + (t.x*t.x)/(xw*xw) + (t.z*t.z)/(zw*zw))) {
        material = _material;
        orig = t;
        height = _height;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0 && t2 <= 0) return hit;
        hit.t = (t2 < t1) ? t2 : t1;
        float tt = (t2 < t1) ? t1 : t2;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        vec4 plane(0, 1, 0, -orig.y-height);
        vec4 currPos4(hit.position.x, hit.position.y, hit.position.z, 1);
        if(dot(plane, currPos4) > 0) {
            hit.t = tt;
            hit.position = ray.start + ray.dir * hit.t;
            hit.normal = normalize(gradf(S + D * hit.t));
            currPos4 = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
        }
        if(dot(plane, currPos4) > 0) {
            hit.t = -1;
            return hit;
        }
        hit.material = material;
        return hit;
    }
};

class Cylinder : public Intersectable, public Quadratics {
    vec3 orig;
    float height;
public:
    Cylinder(vec3 t, float a, float h, Material* _material)
            : Quadratics(
            vec4(1, 0, 0,                        -t.x),
            vec4(0, 0, 0,                        0),
            vec4(0, 0, 1,                        -t.z),
            vec4(-t.x, 0, -t.z, t.x*t.x + t.z*t.z - a*a)) {
        orig = t;
        height = h;
        material = _material;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0 && t2 <= 0) return hit;
        hit.t = (t2 < t1) ? t2 : t1;
        float tt = (t2 < t1) ? t1 : t2;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        vec4 plane1(0, 1, 0, -(orig.y+height/2));
        vec4 plane2(0, 1, 0, -(orig.y-height/2));
        vec4 currPos4(hit.position.x, hit.position.y, hit.position.z, 1);
        if(dot(plane1, currPos4) > 0 || dot(plane2, currPos4) < 0) {
            hit.t = tt;
            hit.position = ray.start + ray.dir * hit.t;
            hit.normal = normalize(gradf(S + D * hit.t));
            currPos4 = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
        }
        if(dot(plane1, currPos4) > 0 || dot(plane2, currPos4) < 0) {
            hit.t = -1;
            return hit;
        }
        hit.material = material;
        return hit;
    }
};

float holeR = 0.3F;
float holeY = 1;

class Ellipsoid : public Intersectable, public Quadratics {
public:
    Ellipsoid(vec3 t, vec3 a, Material* _material)
            : Quadratics(
            vec4(   1/(a.x*a.x),             0,             0, -t.x/(a.x*a.x)),
            vec4(             0,   1/(a.y*a.y),             0, -t.y/(a.y*a.y)),
            vec4(             0,             0,   1/(a.z*a.z), -t.z/(a.z*a.z)),
            vec4(-t.x/(a.x*a.x),-t.y/(a.y*a.y),-t.z/(a.z*a.z), -1 + (t.x*t.x)/(a.x*a.x) + (t.y*t.y)/(a.y*a.y) + (t.z*t.z)/(a.z*a.z))) {
        material = _material;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        hit.material = material;
        return hit;
    }
};

class Hyperboloid : public Intersectable, public Quadratics {
    vec3 orig;
    float height, cut;
public:
    Hyperboloid(vec3 t, float h, float a, float _height, Material* _material, float _cut = 0)
            : Quadratics(
            vec4(   1/(a*a),         0,          0,      -t.x/(a*a)),
            vec4(         0,  -1/(h*h),          0,       t.y/(h*h)),
            vec4(         0,         0,    1/(a*a),      -t.z/(a*a)),
            vec4(-t.x/(a*a), t.y/(h*h), -t.z/(a*a), -1 - (t.y*t.y)/(h*h) + (t.x*t.x)/(a*a) + (t.z*t.z)/(a*a))) {
        material = _material;
        orig = t;
        height = _height;
        cut = _cut;
    }
    Hit intersect(const Ray & ray) override {
        Hit hit;
        vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
        vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float discr = (fv(D, S) + fv(S, D)) * (fv(D, S) + fv(S, D)) - (4 * f(D) * f(S));
        if(discr < 0) return hit;
        float t1 = (-(fv(D, S) + fv(S, D)) + sqrtf(discr)) / (2 * f(D));
        float t2 = (-(fv(D, S) + fv(S, D)) - sqrtf(discr)) / (2 * f(D));
        if(t1 <= 0 && t2 <= 0) return hit;
        hit.t = (t2 < t1) ? t2 : t1;
        float tt = (t2 < t1) ? t1 : t2;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normalize(gradf(S + D * hit.t));
        vec4 plane1(0, 1, 0, -(orig.y+cut+height));
        vec4 plane2(0, 1, 0, -(orig.y+cut));
        vec4 currPos4(hit.position.x, hit.position.y, hit.position.z, 1);
        if((dot(plane1, currPos4) > 0 || dot(plane2, currPos4) < 0)) {
            hit.t = tt;
            hit.position = ray.start + ray.dir * hit.t;
            hit.normal = normalize(gradf(S + D * hit.t));
            currPos4 = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
        }
        if((dot(plane1, currPos4) > 0 || dot(plane2, currPos4) < 0)) {
            hit.t = -1;
            return hit;
        }
        hit.material = material;
        return hit;
    }
};

class Plane : public Intersectable {
    vec3 orig, normal;
public:
    Plane(vec3 _orig, Material* _material, vec3 _normal = vec3(0, 1, 0)) {
        orig = _orig;
        material = _material;
        normal = _normal;
    }
    Hit intersect(const Ray & ray) override {
        // forras: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        Hit hit;
        float div = dot(ray.dir, normal);
        if (fabs(div) > 0.0001f) {
            float t = dot((orig - ray.start), normal) / div;
            if (t > 0) hit.t = t;
        }
        else return hit;
        if(hit.t <= 0) return hit;
        hit.position = ray.start + ray.dir * hit.t;
        if(hit.position.y > holeY && normal.y  >= 0) {
            hit.t = -1;
            return hit;
        }
        if(sqrtf(fabs(hit.position.x)*fabs(hit.position.x) + fabs(hit.position.z)*fabs(hit.position.z)) <= holeR && hit.position.y >= holeY-0.1F) {
            hit.t = -1;
            return hit;
        }
        hit.normal = normal;
        hit.material = material;
        return hit;
    }
};

class Camera {
    vec3 eye, lookAt, up, right;
    float fov;
public:
    Camera(vec3 _eye = vec3(0, 0.2F, -2.5F),
           vec3 _lookAt = vec3(0, 0.2F, -1.5F), float _fov = 45.F) {
        eye = _eye; lookAt = _lookAt; fov = _fov * static_cast<float>(M_PI) / 180;
        vec3 vup(0, 1, 0);
        vec3 w = lookAt - eye;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(const int X, const int Y) {
        vec3 dir = lookAt - eye
                   + right * (2 * static_cast<float>(X) / static_cast<float>(windowWidth) - 1.F)
                   + up * (2 * static_cast<float>(Y) / static_cast<float>(windowHeight) - 1.F);
        return Ray(eye, dir);
    }
};

class Canvas {
    unsigned int vao, textureId;
public:
    Canvas() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    void loadTexture(std::vector<vec4> & image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }
    void draw() {
        glBindVertexArray(vao);
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if(location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

float rnd() { return ((static_cast<float>(rand()) / RAND_MAX) * 2 - 1) * holeR; }

vec2 rndXZ() {
    vec2 res(rnd(), rnd());
    while(length(res) > holeR) {
        res.x = rnd();
        res.y = rnd();
    }
    return res;
}

class Scene {
    std::vector<Intersectable *> objects;
    vec3 sunPos, sun, sky;
    Camera *camera;
    vec3 La;
    const float epsilon = 0.0001F;
    std::vector<vec3> testPoints;
public:
    void build() {
        camera = new Camera();

        for(int i = 0; i < 25; ++i) {
            vec2 rnd = rndXZ();
            testPoints.emplace_back(vec3(rnd.x, holeY, rnd.y));
        }

        sunPos = vec3(4, 3, 0);
        sun = vec3(3, 3, 3);
        sky = vec3(0.32F, 0.5F, 0.8F);
        La = vec3(0.5f, 0.5f, 0.5f);

        vec3 ks(2, 2, 2);
        auto * green = new RoughMaterial(vec3(0.28f, 0.4f, 0.21f), ks, 50);
        auto * blue = new RoughMaterial(vec3(0.17f, 0.19f, 0.4f), ks, 50);
        auto * purple = new RoughMaterial(vec3(0.4f, 0.2f, 0.3f), ks, 50);
        auto * red = new RoughMaterial(vec3(0.4f, 0.17F, 0.11F), ks, 50);
        auto * room = new RoughMaterial(vec3(0.4f, 0.32F, 0.24F), ks, 500);
        // gold
        vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
        auto * gold = new ReflectiveMaterial(n, kappa);
        //silver
        vec3 n_s(0.14f, 0.16f, 0.13f), kappa_s(4.1f, 2.3f, 3.1f);
        auto * silver = new ReflectiveMaterial(n_s, kappa_s);

        objects.push_back(new Plane(vec3(0, -0.5f, 0), blue));
        objects.push_back(new Plane(vec3(0, 0, 4.5F), room, vec3(0, 0, -1)));
        objects.push_back(new Plane(vec3(0, 1, 0), red, vec3(0, -1, 0)));
        objects.push_back(new Plane(vec3(0, 0, -2.6F), green, vec3(0, 0, 1)));
        objects.push_back(new Plane(vec3(-1, 0, 0), room, vec3(1, 0, 0)));
        objects.push_back(new Plane(vec3(1, 0, 0), room, vec3(-1, 0, 0)));
        objects.push_back(new Ellipsoid(vec3(0.5F, -0.5F, -0.3F), vec3(0.25F, 0.6f, 0.25F), gold));
        objects.push_back(new Cylinder(vec3(-0.1F, -0.35F, -0.5F), 0.03f, 0.3F, red));
        objects.push_back(new Paraboloid(vec3(-0.1F, -0.22F, -0.5F), 0.4F, 0.4F, 0.35F, purple));
        objects.push_back(new Ellipsoid(vec3(-0.1F, -0.5F, -0.5F), vec3(0.2F, 0.05F, 0.2F), silver));
        objects.push_back(new Hyperboloid(vec3(-0.6f, -0.05F, -0.2F), 0.1F, 0.05F, 1, green, -0.65F));
        objects.push_back(new Hyperboloid(vec3(0, holeY, 0), 0.6f, holeR, 1.05F, silver));
    }
    void render(std::vector<vec4> & image) {
        for(int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for(int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera->getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }
    vec3 trace(Ray ray, int depth = 0) {
        if(depth > 6) return La;
        Hit hit = firstIntersect(ray);
        if(hit.t < 0) return sky + sun * powf(dot(ray.dir, normalize(sunPos-ray.start)), 10);

        vec3 color(0, 0, 0);
        if(hit.material->type == ROUGH) {
            color = color + hit.material->ka * La;
            vec3 sum(0, 0, 0);
            vec3 holeDir;
            float cosTheta;
            float dOmega;
            unsigned int n = testPoints.size();
            for(const auto & t : testPoints) {
                holeDir = t - (hit.position + hit.normal * epsilon);
                Ray inHole(hit.position + hit.normal * epsilon, holeDir);
                cosTheta = dot(vec3(0, 1, 0), holeDir);
                if(cosTheta > 0 && !shadowIntersect(inHole)) {
                    dOmega = (holeR * holeR * (float)M_PI) / (float)n * cosTheta / (length(holeDir) * length(holeDir));
                    vec3 addition = trace(inHole, depth+1) * dOmega;
                    vec3 halfway = normalize(-ray.dir + holeDir);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0)
                        addition = addition + sun * hit.material->ks * powf(cosDelta, hit.material->shininess) * dOmega;
                    sum = sum + addition;
                }
            }
            color = color + sum;
        } else if(hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.F;
            float cosa = -dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
            color = color + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth+1) * F;
        }
        return color;
    }
    bool shadowIntersect(Ray ray) {
        for (unsigned int i = 0; i < objects.size() - 1; ++i) {
            if (objects[i]->intersect(ray).t > 0)
                return true;
        }
        return false;
    }
    Hit firstIntersect(const Ray & ray) {
        Hit hit;
        Hit test;
        for(const auto & o : objects) {
            test = o->intersect(ray);
            if(test.t > 0 && (hit.t < 0 || test.t < hit.t))
                hit = test;
        }
        if(dot(ray.dir, hit.normal) > 0) hit.normal = hit.normal * (-1);
        return hit;
    }
};

Scene scene;
Canvas *canvas;
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    scene.build();
    canvas = new Canvas();
    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
    canvas->loadTexture(image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0); // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    canvas->draw();

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(const unsigned char key, int /*pX*/, int /*pY*/) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char /*key*/, int /*pX*/, int /*pY*/) {
}

// Move mouse with key pressed
void onMouseMotion(const int pX, const int pY) {
}

// Mouse click event
void onMouse(const int button, const int state, const int pX, const int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

}

